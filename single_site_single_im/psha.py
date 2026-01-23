"""Analytically integrate PSHA to obtain a hazard curve"""

from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp  # For log normal CDF

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


HazardFunction = Callable[
    [npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], HazardMatrix
]


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
    rates = np.asarray(rupture_df[rate_col].values)[:, None]
    means = np.asarray(rupture_df[mean_col].values)[:, None]
    stddevs = np.asarray(rupture_df[stddev_col].values)[:, None]

    hazard_matrix = hazard_function(rates, means, stddevs, threshold_values)
    hazard_vec = hazard_matrix.sum(axis=0)
    return pd.DataFrame(
        {"threshold": threshold_values, "hazard": hazard_vec}
    ).set_index("threshold")
