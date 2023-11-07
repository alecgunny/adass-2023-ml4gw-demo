import time
from collections import defaultdict
from typing import Callable

import h5py
import numpy as np
import torch
from tqdm import tqdm
from tritonclient import grpc as triton

from ml4gw.utils.slicing import unfold_windows


class BatchGenerator(torch.nn.Module):
    def __init__(
        self,
        spectral_density: torch.nn.Module,
        whitener: torch.nn.Module,
        num_channels: int,
        kernel_length: float,
        fduration: float,
        psd_length: float,
        inference_sampling_rate: float,
        sample_rate: float,
    ) -> None:
        super().__init__()
        self.spectral_density = spectral_density
        self.whitener = whitener

        self.step_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.fsize = int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        self.num_channels = num_channels

    @property
    def state_size(self):
        return self.psd_size + self.kernel_size + self.fsize - self.step_size

    def get_initial_state(self):
        return torch.zeros((2, self.num_channels, self.state_size))

    def forward(self, X, state):
        state = torch.cat([state, X], dim=-1)
        split = [self.psd_size, state.size(-1) - self.psd_size]
        whiten_background, X = torch.split(state, split, dim=-1)

        # only use the PSD of the non-injected data for computing
        # our whitening to avoid biasing our PSD estimate
        psd = self.spectral_density(whiten_background[0].double())
        X = self.whitener(X, psd)
        X = unfold_windows(X, self.kernel_size, self.step_size)
        X = X.reshape(-1, self.num_channels, self.kernel_size)
        return X, state[:, :, -self.state_size :]


class TimeSlide:
    def __init__(
        self,
        background: h5py.Group,
        signals: h5py.Group,
        channels: list[str],
        shifts: list[float],
        inference_sampling_rate: float,
        batch_size: int,
    ) -> None:
        self.background = background
        self.signals = signals
        self.channels = channels
        self.shifts = shifts
        self.inference_sampling_rate = inference_sampling_rate

        # infer the sample rate and initial timestamp
        # of the current dataset from data attributes
        self.sample_rate = 1 / background[channels[0]].attrs["dx"]

        # load in the signal timestamps and use
        # the data attributes to convert them to
        # sample points in the timeseries
        self.t0 = background[channels[0]].attrs["x0"]

        # convert a bunch of time units to sample units
        self.step_size = int(self.sample_rate / inference_sampling_rate)
        self.shift_sizes = [int(i * self.sample_rate) for i in shifts]
        self.signal_size = signals[channels[0]].shape[-1]

        # determine the size of the time dimension
        # of the tensor that we'll whiten all at
        # once before turning into a batch
        self.update_size = self.step_size * batch_size
        size = len(background[channels[0]]) - max(self.shift_sizes)
        self.num_batches = int(size // self.update_size)

    def __len__(self):
        return self.num_batches

    @property
    def num_rejected(self):
        return self.signals.attrs["num_rejected"]

    def inject(self, X, offset):
        X_inj = []
        mask = offset <= self.positions
        mask &= self.positions < (offset + self.update_size + self.signal_size)
        idx = np.where(mask)[0]
        pos = self.positions[idx] - offset

        # some fancy array footwork for adding
        # all of our waveforms into our batch
        # at once. Basically create an array
        # of insertion indices for each waveform
        # and then offset them by their position
        # in the target tensor.
        indices = np.arange(-self.signal_size, 0) + 1
        indices = np.repeat(indices[None], len(idx), axis=0)
        indices += pos[:, None]
        indices = indices.reshape(-1)

        underflow = max(-indices.min(), 0)
        overflow = max(indices.max() - self.update_size, 0)
        if underflow or overflow:
            X = np.pad(X, (0, 0, underflow, overflow))
            indices += underflow

        waveforms = [self.signals[i][idx].reshape(-1) for i in self.channels]
        waveforms = np.stack(waveforms)
        X_inj = X.copy()
        X_inj[:, indices] += waveforms
        if underflow or overflow:
            X_inj = X_inj[:, underflow : -overflow or None]
        return X_inj

    def __iter__(self):
        self.times = self.signals["parameters"]["gps_time"][:]
        self.t0 = self.background[self.channels[0]].attrs["x0"]
        diffs = self.times - self.t0
        self.positions = (diffs * self.sample_rate).astype(int)

        for i in range(self.num_batches):
            offset = i * self.update_size
            X = []
            for j, channel in enumerate(self.channels):
                start = self.shift_sizes[j] + offset
                stop = start + self.update_size
                X.append(self.background[channel][start:stop])
            X = np.stack(X)
            X_inj = self.inject(X, offset)
            yield np.stack([X, X_inj])

    def pool(self, x, pool_length):
        pool_size = int(pool_length * self.inference_sampling_rate)
        num_windows = int(len(x) // pool_size)

        size = num_windows * pool_size
        if size < len(x):
            remainder = x[size:]
            x = x[:size]
        else:
            remainder = None

        x = x.reshape(num_windows, pool_size)
        idx = x.argmax(axis=-1)
        offsets = np.arange(num_windows)

        values = x[offsets, idx]
        idx += offsets * pool_size

        if remainder is not None:
            last_idx = remainder.argmax()
            last_val = remainder[last_idx]
            values = np.concatenate([values, [last_val]])
            idx = np.concatenate([idx, [size + last_idx]])
        return values, idx

    def get_events(
        self,
        bg_preds: np.ndarray,
        fg_preds: np.ndarray,
        kernel_length: float,
        psd_length: float,
        fduration: float,
        pool_length: float = 8,
    ):
        drop = int(psd_length * self.inference_sampling_rate)
        drop += int(fduration * self.inference_sampling_rate / 2)
        bg_preds = bg_preds[drop:]
        fg_preds = fg_preds[drop:]

        bg_events, bg_indices = self.pool(bg_preds, pool_length)
        fg_events, fg_indices = self.pool(fg_preds, pool_length)

        bg_times = bg_indices / self.inference_sampling_rate
        bg_times += self.t0 + psd_length + fduration / 2 + kernel_length
        background = dict(
            event_time=bg_times,
            det_stat=bg_events,
            shift=np.ones_like(bg_times)[:, None] * self.shifts,
        )

        # for each known injected event, find the foreground
        # event that is closest to it and call that its
        # detection statistic
        timestamps = self.times - self.t0
        mask = timestamps > (psd_length + fduration / 2)
        mask &= timestamps < len(fg_preds) / self.inference_sampling_rate
        idx = timestamps[mask] * self.inference_sampling_rate

        diffs = np.abs(fg_indices[:, None] - idx)
        argmin = diffs.argmin(axis=0)
        fg_events = fg_events[argmin]
        fg_indices = fg_indices[argmin]

        fg_times = fg_indices / self.inference_sampling_rate
        fg_times += self.t0 + psd_length + fduration / 2 + kernel_length
        foreground = dict(
            event_time=fg_times,
            det_stat=fg_events,
            shift=np.ones_like(fg_times)[:, None] * self.shifts,
        )
        for param, values in self.signals["parameters"].items():
            foreground[param] = values[:][mask]
        return background, foreground


def _cat_dict(d: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate(v, axis=0) for k, v in d.items()}


def infer(
    inference_fn: Callable,
    ifos: list[str],
    kernel_length: float,
    psd_length: float,
    fduration: float,
    inference_sampling_rate: float,
    batch_size: int,
    pool_length: float = 8,
):
    bgf = h5py.File("data/background.hdf5", "r")
    fgf = h5py.File("data/signals.hdf5", "r")

    foreground_events, background_events = defaultdict(list), defaultdict(list)
    with bgf, fgf, torch.no_grad():
        # grab the test split of both files
        bgf, fgf = bgf["test"], fgf["test"]

        # start by generating a bunch of TimeSlide objects up front
        # so we know how much inference we have to do and can track
        # it with a progress bar
        timeslides = []
        for segment in fgf.keys():
            background = bgf[segment]
            foreground = fgf[segment]
            for shifts, signals in foreground.items():
                timeslide = TimeSlide(
                    background,
                    signals,
                    ifos,
                    eval(shifts),
                    inference_sampling_rate,
                    batch_size,
                )
                timeslides.append(timeslide)

        total_steps = sum([len(i) for i in timeslides])
        Tb, num_rejected = 0, 0
        with tqdm(total=total_steps) as pbar:
            # now iterate through each one of these timeslides
            # and generate NN predictions on them
            for timeslide in timeslides:
                # this generates the timeseries of nn outputs
                bg_preds, fg_preds = inference_fn(timeslide, pbar)

                # this turns them into "events": getting rid
                # of duplicates by pooling and aligning
                # injections with their closest event
                bg_events, fg_events = timeslide.get_events(
                    bg_preds,
                    fg_preds,
                    kernel_length,
                    psd_length,
                    fduration,
                    pool_length=pool_length,
                )
                for key, val in bg_events.items():
                    background_events[key].append(val)
                for key, val in fg_events.items():
                    foreground_events[key].append(val)

                Tb += len(bg_preds) / inference_sampling_rate - psd_length
                num_rejected += timeslide.num_rejected

    background_events = _cat_dict(background_events)
    foreground_events = _cat_dict(foreground_events)
    return background_events, foreground_events, num_rejected, Tb


def wait_for_model(model_name, timeout=60):
    client = triton.InferenceServerClient("localhost:8001")
    tick = time.time()
    while (time.time() - tick) < timeout:
        try:
            ready = client.is_model_ready(model_name)
        except Exception:
            time.sleep(1)
        else:
            if ready:
                break
    else:
        raise TimeoutError(f"Model {model_name} never came online")
