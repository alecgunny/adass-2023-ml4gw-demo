from typing import Optional

import h5py
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from tqdm import trange


def project_waveform(
    hp: np.ndarray,
    hc: np.ndarray,
    t_gps: np.ndarray,
    declination: float,
    right_ascension: float,
    polarization: float,
    ifo: str,
) -> np.ndarray:
    detector = Detector(ifo)
    fp, fc = detector.antenna_pattern(
        t_gps=t_gps,
        declination=declination,
        right_ascension=right_ascension,
        polarization=polarization,
    )
    ht = fp * hp + fc * hc
    return ht


def sample_background(
    ifos: list[str],
    sample_size: int,
    psd_size: int,
    dataset: h5py.Group,
    signal_params: Optional[dict[str, np.ndarray]] = None,
):
    load_size = sample_size + psd_size
    size = len(dataset[ifos[0]])
    start = np.random.randint(size - load_size)
    stop = start + load_size

    if signal_params is not None:
        hp, hc = signal_params.pop("plus"), signal_params.pop("cross")
        frac = (start + stop) / 2 / size
        attrs = dataset.attrs
        signal_params["t_gps"] = attrs["start"] + frac * attrs["duration"]

    kernels, backgrounds = {}, {}
    for ifo in ifos:
        x = dataset[ifo][start:stop]
        background, x = np.split(x, [psd_size])
        if signal_params is not None:
            ht = project_waveform(hp, hc, ifo=ifo, **signal_params)
            min_idx = int(sample_size // 2)
            idx = np.random.randint(min_idx, sample_size)
            x[:idx] += ht[-idx:]
        kernels[ifo] = x
        backgrounds[ifo] = background
    return kernels, backgrounds


def make_dataset(
    ifos: list[str],
    background: h5py.Group,
    signal: h5py.Group,
    kernel_length: float,
    fduration: float,
    psd_length: float,
    fftlength: float,
    sample_rate: float,
    highpass: float = 32,
) -> np.ndarray:
    num_waveforms = len(signal["polarizations"]["cross"])
    num_samples = 2 * num_waveforms
    kernel_size = int(kernel_length * sample_rate)
    window_size = kernel_size + int(fduration * sample_rate)
    psd_size = int(psd_length * sample_rate)
    X = np.zeros((num_samples, len(ifos), kernel_size))

    # ensure we're sampling uniformly in time
    datasets = list(background.keys())
    p = [len(background[i][ifos[0]]) for i in datasets]
    p = [i / sum(p) for i in p]

    sky_params = ["declination", "right_ascension", "polarization"]
    sky_params = {i: signal["parameters"][i] for i in sky_params}

    for i in trange(num_samples):
        key = np.random.choice(datasets, p=p)
        dataset = background[key]

        # we'll alternate between background and signals
        idx, rem = divmod(i, 2)
        if rem:
            params = {k: v[idx] for k, v in sky_params.items()}
            params["cross"] = signal["polarizations"]["cross"][idx]
            params["plus"] = signal["polarizations"]["plus"][idx]
        else:
            params = None

        kernels, backgrounds = sample_background(
            ifos, window_size, psd_size, dataset, params
        )
        for j, ifo in enumerate(ifos):
            x = TimeSeries(kernels[ifo], sample_rate=sample_rate)
            bg = TimeSeries(backgrounds[ifo], sample_rate=sample_rate)
            asd = bg.asd(fftlength, window="hann", method="median")
            x = x.whiten(asd=asd, fduration=fduration, highpass=highpass)
            x = x.crop(0.5, 1.5)
            X[i, j] = x.value

    y = np.zeros((num_samples, 1))
    y[1::2] = 1
    return X, y
