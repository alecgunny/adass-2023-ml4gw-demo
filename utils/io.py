import h5py
import numpy as np


def read_timeseries(fname, channels, start, duration, x=None):
    offset, size = None, None
    with h5py.File(fname, "r") as f:
        for i, channel in enumerate(channels):
            dataset = f[channel]
            if offset is None:
                sample_rate = 1 / dataset.attrs["dx"]
                offset = int(start * sample_rate)
                size = int(duration * sample_rate)
                if x is None:
                    x = np.zeros((len(channels), size))
            x[i] = dataset[offset : offset + size]
    return x
