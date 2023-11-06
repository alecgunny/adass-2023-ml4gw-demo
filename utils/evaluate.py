import numpy as np
from astropy.cosmology import Planck15 as cosmology
from bilby.core.prior import Constraint, LogNormal, PriorDict
from scipy.integrate import quad


def mass_ratio(params):
    params["mass_ratio"] = params["mass2"] / params["mass1"]
    return params


def volume_element(z):
    return cosmology.differential_comoving_volume(z).value / (1 + z)


def get_astrophysical_volume(zmax=2):
    volume, _ = quad(volume_element, 0, zmax)
    return volume * 4 * np.pi / 10**9


def lognorm_prob(x, median, sigma):
    mu = np.log(median)
    exp = -((np.log(x) - mu) ** 2) / sigma**2 / 2
    norm = 1 / x / sigma / (2 * np.pi) ** 0.5
    return norm * np.exp(exp)


def evaluate(
    background_events: dict[str, np.ndarray],
    foreground_events: dict[str, np.ndarray],
    num_rejected: int,
    livetime: float,
    max_far_per_week: 100,
):
    seconds_per_week = 3600 * 24 * 7
    max_events = int(max_far_per_week * livetime / seconds_per_week)
    thresholds = np.sort(background_events["det_stat"])[-max_events:][::-1]
    fars = seconds_per_week * np.arange(1, max_events + 1) / livetime

    mass_combos = [(35, 35), (35, 20), (20, 20), (20, 10)]
    num_accepted = len(foreground_events["det_stat"])
    weights = np.zeros((len(mass_combos), num_accepted, 1))

    mass_dict = {
        k: v for k, v in foreground_events.items() if k.startswith("mass")
    }
    for i, (m1, m2) in enumerate(mass_combos):
        prior = dict(
            mass1=LogNormal(mu=np.log(m1), sigma=0.1),
            mass2=LogNormal(mu=np.log(m2), sigma=0.1),
            mass_ratio=Constraint(0.02, 1),
        )
        prior = PriorDict(prior, conversion_function=mass_ratio)
        prob = prior.prob(mass_dict, axis=0)
        weights[i, :, 0] = prob / foreground_events["prob"]

    mask = foreground_events["det_stat"][:, None] >= thresholds
    detections = weights * mask
    total_events = num_accepted + num_rejected
    detections = detections.sum(axis=1) / total_events

    volume = get_astrophysical_volume(zmax=2)
    detections *= volume
    return fars, dict(zip(mass_combos, detections)), weights
