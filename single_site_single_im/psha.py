"""Analytically integrate PSHA to obtain a hazard curve"""

import functools
from typing import Any, Callable, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp  # For log normal CDF
import tqdm

HazardMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]


def analytical_hazard(
    rate: npt.ArrayLike,
    log_im_mean: npt.ArrayLike,
    log_im_stddev: npt.ArrayLike,
    threshold: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    r"""Compute rupture hazard using analytical result.

    Assumes intensity measure is log-normally distributed with ``s=log_im_stddev`` and ``scale=exp(log_im_mean)``.
    Computes hazard as $\text{rate} * (1 - F(\text{threshold}))$ where $F$ is the CDF of the IM distribution.


    Parameters
    ----------
    rate : ArrayLike
       Rupture rate(s) for the sources.
    log_im_mean : ArrayLike
       Log of the mean intensity measure value.
    log_im_stddev : ArrayLike
       Log of the intensity measure std. deviation.
    threshold : ArrayLike
       Threshold values of the intensity measure to compute hazard
       for. For example, if the intensity measure is PGA, then
       threshold may take a value of 0.65g.

    Returns
    -------
    ArrayLike
        Hazard for each rate, intensity measure distribution, and
        threshold value given.

    See Also
    --------
    sp.stats.lognorm : For the description of the ``s`` and ``scale`` parameters.
    """

    rate = np.asarray(rate)
    log_im_mean = np.asarray(log_im_mean)
    log_im_stddev = np.asarray(log_im_stddev)
    threshold = np.asarray(threshold)

    return rate * sp.stats.lognorm.sf(
        threshold, s=log_im_stddev, scale=np.exp(log_im_mean)
    )


def monte_carlo_rupture_hazard(
    log_im_mean: npt.ArrayLike,
    log_im_stddev: npt.ArrayLike,
    threshold: npt.ArrayLike,
    samples: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    r"""Compute rupture hazard using Naive Monte Carlo method.

    Assumes intensity measure is log-normally distributed with ``s=log_im_stddev`` and ``scale=exp(log_im_mean)``.
    Computes hazard using the estimator

    $$ \frac{1}{N} \sum_{i = 1}^N I(X_i > \text{threshold}),$$

    where $X_i$ is a ground motion sampled from the log-normal
    intensity measure distribution and $N$ is the number of samples
    per rupture.


    Parameters
    ----------
    rate : ArrayLike
       Rupture rate(s) for the sources.
    log_im_mean : ArrayLike
       Log of the mean intensity measure value.
    log_im_stddev : ArrayLike
       Log of the intensity measure std. deviation.
    threshold : ArrayLike
       Threshold values of the intensity measure to compute hazard
       for. For example, if the intensity measure is PGA, then
       threshold may take a value of 0.65g.
    samples : ArrayLike
       Number of samples for each rupture. Used to estimate hazard for each individual rupture.
    weights : ArrayLike
       Weights for each rupture. Used for importance sampling.

    Returns
    -------
    ArrayLike
        Hazard for each rate, intensity measure distribution, and
        threshold value given.

    See Also
    --------
    sp.stats.lognorm : For the description of the ``s`` and ``scale`` parameters.
    """

    log_im_mean = np.asarray(log_im_mean)
    log_im_stddev = np.asarray(log_im_stddev)
    threshold = np.asarray(threshold)
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    n_rup = len(samples)
    n_thresh = len(threshold)
    hazard_matrix = np.zeros((n_rup, n_thresh), np.float64)
    for i, (n, weight, rup_im_mean, rup_im_stddev) in enumerate(
        zip(samples, weights, log_im_mean, log_im_stddev)
    ):
        if n == 0:
            continue
        im_samples = sp.stats.lognorm.rvs(
            s=rup_im_stddev, scale=np.exp(rup_im_mean), size=n
        )
        hazard_matrix[i] = weight * (threshold[:, np.newaxis] <= im_samples).sum(axis=1)

    return hazard_matrix


HazardFunction = Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], HazardMatrix]


def calculate_hazard(
    rupture_df: pd.DataFrame,
    threshold_values: npt.NDArray[np.floating],
    hazard_function: HazardFunction,
    rate_col: str = "rate",
    mean_col: str = "PGA_mean",
    stddev_col: str = "PGA_std_Total",
) -> pd.DataFrame:
    """Simulate and aggregate hazard over a number of ruptures.

    Takes in a rupture dataframe describing the intensity measure
    distribution and some threshold values. Uses ``rupture_hazard`` to
    estimate the probability of threshold exceedence at
    ``threshold_values`` using ``hazard_function`` and aggregates the
    results over all ruptures.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        The dataframe of ruptures. Must have ``rate``, ``PGA_mean``
        and ``PGA_std_Total`` columns.
    threshold_values : npt.NDArray[np.floating]
        Threshold values for test for.
    hazard_function : HazardFunction
        Hazard function to estimate each hazard for each rupture. Must
        accept an array of rates, means and stddev with shape (N_rup,
        1) each. Must return hazard in the shape (N_rup, N_thresh).
    rate_col : str, optional
        The rate column to extract from ``rupture_df``.
    mean_col : str, optional
        The mean column to extract from ``rupture_df``.
    stddev_col : str, optional
        The stddev column to extract from ``rupture_df``.

    Returns
    -------
    pd.DataFrame
        A dataframe of hazard values aggregated over each rupture
        (i.e. a hazard curve). Indexed by ``threshold_values``. The
        ``hazard`` column contains hazard results.
    """
    means = np.asarray(rupture_df[mean_col].values)[:, None]
    stddevs = np.asarray(rupture_df[stddev_col].values)[:, None]

    hazard_matrix = hazard_function(means, stddevs, threshold_values)
    hazard_vec = hazard_matrix.sum(axis=0)
    return pd.DataFrame(
        {"threshold": threshold_values, "hazard": hazard_vec}
    ).set_index("threshold")


class BootstrapResult(NamedTuple):
    ci_low: np.ndarray[tuple[int,], np.dtype[np.float64]]
    ci_high: np.ndarray[tuple[int,], np.dtype[np.float64]]
    variance: np.ndarray[tuple[int,], np.dtype[np.float64]]
    samples: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    thresholds: np.ndarray[tuple[int,], np.dtype[np.float64]]


def bootstrap_sampling_strategy(
    strategy: Callable[[pd.DataFrame], pd.DataFrame],
    rupture_df: pd.DataFrame,
    threshold_values: npt.ndarray[tuple[int,], np.dtype[np.float64]],
    n_resamples: int = 1000,
    **kwargs: Any,
) -> BootstrapResult:
    """Bootstrap a rupture sampling strategy.



    Parameters
    ----------
    strategy : pd.DataFrame
        The rupture sampling plan, must have a ``samples`` column, may
        have a ``weights`` column for importance or stratified
        sampling.


    Returns
    -------
    pd.DataFrame
        Hazard curves bootstrapped from the sampling strategy.
    """

    estimates = []
    for _ in tqdm.trange(n_resamples, unit="samples", **kwargs):
        strategy_df = strategy(rupture_df)
        weights = (
            strategy_df["weights"]
            if "weights" in strategy_df.columns
            else np.ones_like(strategy_df["samples"])
        )
        estimates.append(
            np.asarray(
                calculate_hazard(
                    rupture_df,
                    threshold_values,
                    hazard_function=functools.partial(
                        monte_carlo_rupture_hazard,
                        samples=strategy_df["samples"],
                        weights=weights,
                    ),
                )["hazard"].values
            )
        )
    estimates = np.stack(
        estimates,
        axis=1,
    )
    ci_low = np.percentile(estimates, 5, axis=-1)
    ci_high = np.percentile(estimates, 95, axis=-1)
    variance = np.var(estimates, axis=1)
    return BootstrapResult(
        ci_low=ci_low,
        ci_high=ci_high,
        variance=variance,
        samples=estimates,
        thresholds=threshold_values,
    )


def naive_monte_carlo_sampling_strategy(ruptures: pd.DataFrame, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"samples": np.full_like(ruptures.index, n), "weights": ruptures["rate"] / n},
        index=ruptures.index,
    )


def poisson_catalogue_sampling_strategy(
    ruptures: pd.DataFrame, length: float
) -> pd.DataFrame:
    samples = sp.stats.poisson(mu=ruptures["rate"] * length).rvs()
    weights = np.full(len(ruptures), 1 / length)
    return pd.DataFrame(
        {
            "samples": samples,
            "weights": weights,
        },
        index=ruptures.index,
    )


def poisson_mean_ruptures_sampled(ruptures: pd.DataFrame, length: float) -> int:
    return round(ruptures["rate"].sum() * length)
