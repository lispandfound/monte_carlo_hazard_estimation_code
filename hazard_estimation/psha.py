"""Analytically integrate PSHA to obtain a hazard curve"""

import functools
from dataclasses import dataclass
from typing import Callable, NamedTuple, Protocol

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp  # For log normal CDF
import tqdm
from numpy.random import Generator

from hazard_estimation.site import Site

Array1 = np.ndarray[tuple[int,], np.dtype[np.float64]]
IArray1 = np.ndarray[tuple[int,], np.dtype[np.int64]]
HazardMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]


@dataclass
class SourceModel:
    """Clean arrays extracted from the raw dataframe."""

    rates: Array1
    mean_magnitudes: Array1
    stddev_magnitudes: Array1

    def __post_init__(self):
        self.rates = np.asarray(self.rates)
        self.mean_magnitudes = np.asarray(self.mean_magnitudes)
        self.stddev_magnitudes = np.asarray(self.stddev_magnitudes)


@dataclass
class SourceToSite:
    """Source-to-site distance parameters."""

    rrup: Array1

    def __post_init__(self):
        self.rrup = np.asarray(self.rrup)


@dataclass
class SimulationPlan:
    """The instruction set for the simulation kernel."""

    counts: IArray1
    weights: Array1

    def __post_init__(self):
        self.counts = np.asarray(self.counts)
        self.weights = np.asarray(self.weights)


class SamplingStrategy(Protocol):
    def __call__(
        self, sources: pd.DataFrame, rng: Generator
    ) -> SimulationPlan: ...  # numpydoc ignore=GL08


STDDEV_UPPER = 1
STDDEV_LOWER = -1


def get_leonard_magnitude_params(area: Array1, rake: Array1) -> tuple[Array1, Array1]:
    """Leonard (2014) magnitude scaling parameters.

    Parameters
    ----------
    area : Array1
        Area of the fault (km^2).
    rake : Array1
        Rake of the fault (degrees).

    Returns
    -------
    tuple[Array1, Array1]
        (mean_magnitude, stddev_magnitude)
    """
    abs_rake = np.abs(rake)
    is_strike_slip = (abs_rake <= 45) | (abs_rake >= 135)

    # Strike-slip: mu=3.99, sigma=0.26 | Others: mu=4.03, sigma=0.30
    mu_constant = np.where(is_strike_slip, 3.99, 4.03)
    sigma = np.where(is_strike_slip, 0.26, 0.30)

    mean_magnitude = np.log10(area) + mu_constant

    return mean_magnitude, sigma


def analytical_hazard(
    source_model: SourceModel,
    site: Site,
    source_to_site: SourceToSite,
    threshold: float,
) -> npt.NDArray[np.floating]:
    r"""Compute rupture hazard using analytical result.

    Assumes intensity measure is log-normally distributed with ``s=log_im_stddev`` and ``scale=exp(log_im_mean)``.
    Computes hazard as $\text{rate} * (1 - F(\text{threshold}))$ where $F$ is the CDF of the IM distribution.


    Parameters
    ----------


    Returns
    -------
    ArrayLike
        Hazard for each rate, intensity measure distribution, and
        threshold value given.

    See Also
    --------
    sp.stats.lognorm : For the description of the ``s`` and ``scale`` parameters.
    """
    std_norm = sp.stats.norm()
    z = np.linspace(-1, 1)
    phi_z = std_norm.pdf(z)
    normalising_constant = std_norm.cdf(STDDEV_UPPER) - std_norm.cdf(STDDEV_LOWER)
    mags = (
        source_model.mean_magnitudes[:, None]
        + z[None, :] * source_model.stddev_magnitudes[:, None]
    )
    rupture_df = pd.DataFrame(
        {
            "vs30": site.vs30,
            "mag": mags.flatten(),
            "rrup": np.repeat(
                source_to_site.rrup, mags.size // source_to_site.rrup.size
            ),
        }
    )

    import oq_wrapper as oqw

    gmm_outputs = oqw.run_gmm(
        oqw.constants.GMM.A_22, oqw.constants.TectType.ACTIVE_SHALLOW, rupture_df, "PGA"
    )
    log_mean = gmm_outputs["PGA_mean"].values.reshape(mags.shape)
    log_stddev = gmm_outputs["PGA_std_Total"].values.reshape(mags.shape)
    cond_prob = sp.stats.lognorm.sf(threshold, s=log_stddev, scale=np.exp(log_mean))
    integrand = cond_prob * phi_z[None, :]
    total_exceedence = np.trapezoid(integrand, x=z, axis=1)
    total_exceedence /= normalising_constant
    return source_model.rates * total_exceedence


@numba.njit(cache=True)
def _threshold_reduction(
    ground_motions: np.ndarray, samples: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    # Initialize with the number of ruptures (len(samples))
    hazard_matrix = np.zeros((len(samples), len(thresholds)), dtype=np.int64)

    idx = 0
    for k, sample_count in enumerate(samples):
        if sample_count == 0:
            continue

        # Iterate through the block of ground motions for this rupture
        for i in range(idx, idx + sample_count):
            gm = ground_motions[i]  # Cache the value
            for j in range(len(thresholds)):
                if gm >= thresholds[j]:
                    hazard_matrix[k, j] += 1  # Use k (rupture index), not i

        idx += sample_count

    return hazard_matrix.astype(np.float64)


def monte_carlo_rupture_hazard(
    model: SourceModel, plan: SimulationPlan, thresholds: Array1, rng: Generator
) -> HazardMatrix:
    r"""Compute rupture hazard using Naive Monte Carlo method.

    Assumes intensity measure is log-normally distributed with ``s=log_im_stddev`` and ``scale=exp(log_im_mean)``.
    Computes hazard using the estimator

    $$ \frac{1}{N} \sum_{i = 1}^N I(X_i > \text{threshold}),$$

    where $X_i$ is a ground motion sampled from the log-normal
    intensity measure distribution and $N$ is the number of samples
    per rupture.


    Parameters
    ----------
    model : SourceModel
        Empirical ground motion model for each rupture to draw samples from.
    plan : SimulationPlan
        Sampling strategy and weights for Monte Carlo trials
    thresholds : array of floats
        Threshold values to sample from.
    rng : Generator
        Random number generator for ground motion samples.

    Returns
    -------
    array of floats
        Aggregate hazard for each threshold value given.
    """
    means_flat = np.repeat(model.log_means, plan.counts)
    stds_flat = np.repeat(model.log_stds, plan.counts)

    ground_motions = sp.stats.lognorm.rvs(
        s=stds_flat, scale=np.exp(means_flat), random_state=rng
    )

    raw_matrix = _threshold_reduction(ground_motions, plan.counts, thresholds)
    weighted_hazard = raw_matrix * plan.weights[:, np.newaxis]
    return weighted_hazard.sum(axis=0)


class BootstrapResult(NamedTuple):
    """Namedtuple containing bootstrap results."""

    ci_low: np.ndarray[tuple[int,], np.dtype[np.float64]]
    ci_high: np.ndarray[tuple[int,], np.dtype[np.float64]]
    mean: np.ndarray[tuple[int,], np.dtype[np.float64]]
    variance: np.ndarray[tuple[int,], np.dtype[np.float64]]
    samples: np.ndarray[tuple[int, int], np.dtype[np.float64]]


def run_bootstrap(
    simulation_fn: Callable[[], np.ndarray],
    n_resamples: int,
    use_tqdm: bool = True,
    tqdm_desc: str = "Bootstrapping",
) -> BootstrapResult:
    """
    Pure orchestration.
    simulation_fn is a pre-configured closure that returns a single hazard curve.
    """
    results = []
    iter = tqdm.trange(n_resamples, desc=tqdm_desc) if use_tqdm else range(n_resamples)
    for _ in iter:
        results.append(simulation_fn())

    samples = np.stack(results, axis=0)  # Shape: (N_boot, N_thresh)
    return BootstrapResult(
        mean=np.mean(samples, axis=0),
        ci_low=np.percentile(samples, 2.5, axis=0),
        ci_high=np.percentile(samples, 97.5, axis=0),
        variance=np.var(samples, axis=0),
        samples=samples,
    )


def naive_strategy(ruptures: pd.DataFrame, n: int) -> SimulationPlan:
    """Naive monte carlo strategy where every rupture is sampled an equal number of times.

    Parameters
    ----------
    ruptures : pd.DataFrame
        Ruptures to sample.
    n : int
        The number of samples per rupture.

    Returns
    -------
    SimulationPlan
        A naive simulation strategy sampling every rupture exactly ``n`` times.
    """
    return SimulationPlan(
        counts=np.full(len(ruptures), n, dtype=np.int64),
        weights=np.full(len(ruptures), ruptures["rate"] / n, dtype=np.float64),
    )


def poisson_strategy(df: pd.DataFrame, rng: Generator, length: float) -> SimulationPlan:
    """Poisson catalogue sampling strategy.

    Sampling strategy implemented from [0]_.

    Parameters
    ----------
    ruptures : pd.DataFrame
        Ruptures to sample.
    length : float
        Length of synthetic catalogue (in years).

    Returns
    -------
    pd.DataFrame
        A strategy dataframe.

    Reference
    ---------
    .. [0] Azar, S., & Dabaghi, M. (2021). Simulation-Based Seismic
    Hazard Assessment Using Monte-Carlo Earthquake Catalogs:
    Application to CyberShake. Bulletin of the Seismological Society
    of America, 111(3), 1481–1493. https://doi.org/10.1785/0120200375
    """
    rates = np.asarray(df["rate"].values)
    counts = sp.stats.poisson.rvs(rates * length, random_state=rng).astype(np.int64)
    weights = np.full(len(df), 1.0 / length)
    return SimulationPlan(counts=counts, weights=weights)


def poisson_mean_ruptures_sampled(ruptures: pd.DataFrame, length: float) -> int:
    """Counts the average number of sampled ruptures the poisson strategy produces for a given length.

    Parameters
    ----------
    ruptures : pd.DataFrame
        Ruptures that would be sampled.
    length : float
        Length of the synthetic catalogue (in years).

    Returns
    -------
    int
        The expected number of ruptures, rounded to the nearest
        rupture.

    See Also
    --------
    poisson_catalogue_sampling_strategy : The sampling strategy this counts the mean samples for.
    """
    return round(ruptures["rate"].sum() * length)


def cybershake_nz_strategy(ruptures: pd.DataFrame) -> SimulationPlan:
    """Cybershake New Zealand baseline rupture strategy.

    Samples ruptures proportional to their magnitude. Higher
    magnitudes are sampled more often. Ruptures below magnitude 6 and
    above magnitude 8 are clipped to 14 and 68 samples respectively.

    Parameters
    ----------
    ruptures : pd.DataFrame
        Ruptures that would be sampled.

    Returns
    -------
    SimulationPlan
        A simulation plan implementing the Cybershake NZ sampling method.
    """
    magnitude = ruptures["mag"]
    counts = np.round(np.clip(27 * magnitude - 148, 14, 68)).astype(np.int64)
    weights = ruptures["rate"] / counts
    return SimulationPlan(counts=counts, weights=weights)


SCEC_SAMPLING_SPACE = 4.0


def scec_cybershake_strategy(
    ruptures: pd.DataFrame, hypocentre_spacing: float
) -> SimulationPlan:
    """SCEC Cybershake sampling strategy [0]_.

    Samples ruptures proportional to area. Larger faults are sampled more often.

    Parameters
    ----------
    ruptures : pd.DataFrame
        Ruptures to sample.
    hypocentre_spacing : float
        Spacing for hypocentre distribution.

    Returns
    -------
    pd.DataFrame
        A sampling strategy.

    Notes
    -----
    Exact formula is ``A / (H^2)`` where ``A`` is area and ``H``
    hypocentre spacing. The default hypocentre spacing is set to 4km
    after personal communication with Scott Callaghan.

    References
    ----------
    .. [0] Rupture Variation Generator V5.4.2 - SCECpedia. (n.d.). Retrieved January 29, 2026, from https://strike.scec.org/scecpedia/Rupture_Variation_Generator_v5.4.2#target_hypo_spacing

    See Also
    --------
    SCEC_SAMPLING_SPACE : The SCEC default sample spacing.
    """
    area = ruptures["area"]
    # Hypocentres in the SCEC sampling strategy are placed every 4km
    # along-strike and down-dip. This means l/4 * w/4 = lw / 16
    # samples. Note length * width != area in general due to shearing
    # of fault planes when dip dir is not strike + 90. However, it is
    # a good enough approximation that I will use area as a stand-in
    # for length * width here.
    counts = (area / (hypocentre_spacing * hypocentre_spacing) + 0.5).astype(np.int64)
    weights = ruptures["rate"] / counts
    return SimulationPlan(counts=counts, weights=weights)


def fixed_effort_poisson_strategy(
    ruptures: pd.DataFrame, rng: Generator, n: int
) -> SimulationPlan:
    probabilities = ruptures["rate"] / ruptures["rate"].sum()
    return importance_sampled_strategy(ruptures, probabilities, rng, n)


def importance_sampled_strategy(
    ruptures: pd.DataFrame,
    rupture_prob_distribution: pd.Series,
    rng: Generator,
    n: int,
) -> SimulationPlan:
    rupture_prob_distribution = rupture_prob_distribution.loc[ruptures.index]
    counts = rng.multinomial(n, rupture_prob_distribution)
    weights = (
        ruptures["rate"]
        .div(rupture_prob_distribution * n)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    return SimulationPlan(counts=counts, weights=weights)


def optimal_allocation_naive_strategy(ruptures: pd.DataFrame, n: int) -> SimulationPlan:
    c = np.sqrt(ruptures["hazard"] * (1 - ruptures["hazard"] / ruptures["rate"]))
    sample_weight = c / c.sum()
    samples = np.round(n * sample_weight)
    weights = ruptures["rate"] * np.where(samples != 0, 1 / samples, 0)
    return SimulationPlan(counts=samples.astype(np.int64), weights=weights)


def rate_allocation_naive_strategy(ruptures: pd.DataFrame, n: int) -> SimulationPlan:
    sample_weight = ruptures["rate"] / ruptures["rate"].sum()
    samples = np.round(n * sample_weight)
    weights = ruptures["rate"] * np.where(samples > 0, 1 / samples, 0)
    return SimulationPlan(counts=samples.astype(np.int64), weights=weights)
