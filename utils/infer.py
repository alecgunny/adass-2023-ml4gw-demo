import h5py
import torch
from tqdm import trange

from ml4gw.utils.slicing import unfold_windows


@torch.no_grad()
def infer_on_timeslide(
    model: torch.nn.Module,
    background: h5py.Group,
    signals: h5py.Group,
    channels: list[str],
    shifts: list[float],
    kernel_length: float,
    inference_sampling_rate: float,
    fduration: float,
    psd_length: float,
    batch_size: int
):
    # infer the sample rate and initial timestamp
    # of the current dataset from data attributes
    sample_rate = 1 / background[channels[0]].attrs["dx"]
    t0 = background[channels[0]].attrs["x0"]

    # load in the signal timestamps and use
    # the data attributes to convert them to
    # sample points in the timeseries
    times = signals["parameters"]["gps_time"][:] - t0
    positions = (times * sample_rate).astype(int)
    positions = torch.from_numpy(positions).to("cuda")
    signal_size = signals[channels[0]].shape[-1]

    # convert a bunch of time units to sample units
    step_size = int(sample_rate / inference_sampling_rate)
    shift_sizes = [int(i * sample_rate) for i in shifts]
    psd_size = int(psd_length * sample_rate)
    kernel_size = int(kernel_length * sample_rate)
    fsize = int(fduration * sample_rate)

    # determine the size of the time dimension
    # of the tensor that we'll whiten all at
    # once before turning into a batch
    xsize = step_size * (batch_size - 1) + kernel_size + fsize
    update_size = batch_size * step_size
    full_size = psd_size + xsize
    state_size = full_size - update_size
    size = len(background[channels[0]])
    size -= max(shift_sizes) + state_size + step_size

    # compute how many complete batches we anticipate going through
    num_steps = int(size // step_size) + 1
    num_batches = int(num_steps // batch_size)

    # initialize the state with pure background data
    # for computing the first PSDs
    snapshot = torch.Tensor((len(channels)), state_size)
    for i, channel in enumerate(channels):
        start = shift_sizes[i]
        stop = start + state_size
        snapshot[i] = torch.Tensor(background[channel][start: stop])
    snapshot = snapshot.to("cuda")

    # initialize a blank tensor of updates that
    # we'll fill out with new data at each step
    update_size = batch_size * step_size
    update = torch.zeros((len(channels), update_size), device="cuda")

    bg_preds, fg_preds = [], []
    for i in trange(num_batches):
        # this is the offset to the first sample
        # of the timeseries we'll whiten, batch,
        # then do inference on
        offset = psd_size + i * update_size
        for j, channel in enumerate(channels):
            # offset the start index by the channel's shift
            # as well as (xsize - update) to account for
            # the difference between the psd size and the
            # state size. If your head hurts, just imagine
            # writing this.
            start = shift_sizes[j] + offset - xsize + update_size
            stop = start + update.size(-1)
            x = torch.Tensor(background[channel][start: stop]).to("cuda")
            update[j] = x

        # append our updates to our snapshot, slough off old
        # data to create a new snaphsot, then split the
        # combined snapshot/update into a background for
        # PSD calculation and a timeseries to be whitened
        X = torch.cat([snapshot, update], dim=1)
        snapshot = X[:, -state_size:]
        whiten_background, X = torch.split(X, [psd_size, xsize], dim=1)

        # do our injections. In principle this should live
        # in its own function, or even better in its own
        # class (see the aframe repo), but I'm going to
        # write this all out here to (1) be very clear
        # as to what's happening and (2) show that making
        # optimized code can be tricky but is ultimately
        # about minimizing data transfer and total number
        # of memory-bound ops like slicing or simple
        # array algebra (which has a low compute/memory ratio).
        X_inj = []
        mask = offset <= positions
        mask &= positions < (offset + xsize + signal_size)
        idx = torch.where(mask)[0]
        idx_np = idx.cpu().numpy()
        pos = positions[idx] - offset

        # some fancy array footwork for adding
        # all of our waveforms into our batch
        # at once. Basically create an array
        # of insertion indices for each waveform
        # and then offset them by their position
        # in the target tensor.
        indices = torch.arange(-signal_size, 0) + 1
        indices = indices.to(X.device)
        indices = indices.view(1, -1).repeat(len(idx), 1)
        indices += pos[:, None]
        indices = indices.view(-1)
        for j, channel in enumerate(channels):
            # load in the waveforms that have some signal
            # in this timeseries, and offset our global
            # indices by this channel's shift.
            waveforms = torch.Tensor(signals[channel][idx_np]).to(X)
            inject_idx = indices - shift_sizes[j]

            # some of our waveforms may extend beyond the
            # current batch. In this case, the fastest
            # thing to do will just be to pad some zeros,
            # inject everything, then slice off the data
            # that we don't need. It ain't pretty but
            # that's showbiz baby.
            underflow = int(max(-inject_idx.min().item(), 0))
            overflow = int(max(inject_idx.max().item() - xsize, 0))
            x = X[j] + 0
            if underflow or overflow:
                x = torch.nn.functional.pad(x, (underflow, overflow))
                inject_idx += underflow

            x[inject_idx] += waveforms.view(-1)
            if underflow or overflow:
                x = x[underflow: -overflow or None]
            X_inj.append(x)
        X_inj = torch.stack(X_inj)

        # we'll just use our ml4gw objects here
        psd = model.spectral_density(whiten_background.double())
        X = model.whitener(X, psd)[0]
        X_inj = model.whitener(X_inj, psd)[0]

        # unfold into batches
        X = unfold_windows(X, kernel_size, step_size)
        X_inj = unfold_windows(X_inj, kernel_size, step_size)

        # # do inference
        y_bg = model(X)[:, 0]
        y_fg = model(X_inj)[:, 0]
        bg_preds.append(y_bg)
        fg_preds.append(y_fg)

    bg_preds = torch.cat(bg_preds)
    fg_preds = torch.cat(fg_preds)
    return bg_preds, fg_preds