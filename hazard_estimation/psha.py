"""Analytically integrate PSHA to obtain a hazard curve"""

import math
from pathlib import Path

import cyclopts
import geopandas as gpd
import numba
import numpy as np
import numpy.typing as npt
import oq_wrapper as oqw
import pandas as pd
import scipy as sp
import tqdm
import xarray as xr
from numba_stats import truncnorm

Array1 = np.ndarray[tuple[int,], np.dtype[np.float64]]
IArray1 = np.ndarray[tuple[int,], np.dtype[np.int64]]
HazardMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]

app = cyclopts.App()

STDDEV_UPPER = 1.0
STDDEV_LOWER = -1.0


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


def monte_carlo_sample(
    ruptures: pd.DataFrame, n: int, z_lower: float, z_upper: float, column: str
) -> xr.DataArray:
    sample = ruptures.sample(n, replace=True, weights=column)
    mean_magnitude, sigma = get_leonard_magnitude_params(
        sample["area"] / 1e6, sample["rake"]
    )
    magnitudes = sp.stats.truncnorm.rvs(
        z_lower, z_upper, loc=mean_magnitude, scale=sigma
    )
    return xr.DataArray(
        magnitudes, dims=("rupture",), coords=dict(rupture=sample.index.values)
    )


def analytical_rupture_sample(
    ruptures: pd.DataFrame,
    z_lower: float = STDDEV_LOWER,
    z_upper: float = STDDEV_UPPER,
    num: int = 50,
) -> xr.DataArray:
    mean_magnitude, sigma = get_leonard_magnitude_params(
        ruptures["area"].values / 1e6, ruptures["rake"]
    )
    z = np.linspace(z_lower, z_upper, num=num)
    #           (N_r, N_z)   +            (Nr, Nz) * (Nr, Nz)
    magnitude = mean_magnitude[:, None] + z[None, :] * sigma[:, None]
    return xr.DataArray(
        magnitude,
        dims=("rupture", "z"),
        coords=dict(z=z, rupture=ruptures.index.values),
    )


def rupture_distances(source_to_site: pd.DataFrame) -> xr.DataArray:
    source_to_site = source_to_site.reset_index().sort_values(["site", "rupture"])
    n_ruptures = len(source_to_site["rupture"].unique())
    rrup = source_to_site["rrup"].values
    sites = source_to_site["site"].unique()
    ruptures = source_to_site["rupture"].values[:n_ruptures]
    rrup = rrup.reshape((-1, n_ruptures))
    return xr.DataArray(
        rrup, dims=("site", "rupture"), coords=dict(site=sites, rupture=ruptures)
    )


def ground_motion_inputs(
    rupture_parameters: xr.DataArray,
    rupture_distances: xr.DataArray,
    sites: pd.DataFrame,
) -> xr.Dataset:
    vs30 = sites["vs30"].to_xarray()
    dset = xr.Dataset(
        dict(
            rrup=rupture_distances.sel(
                rupture=rupture_parameters.rupture.values, site=vs30.site.values
            ),
            mag=rupture_parameters,
            vs30=vs30,
        )
    )

    return dset.stack(realisation=dset.dims)


def run_ground_motion_model(
    ground_motion_inputs: xr.Dataset, intensity_measure: str, period: float
) -> xr.Dataset:
    # Final flat array for vectorised gmm:
    gmm_inputs = ground_motion_inputs.to_dataframe()
    gmm_outputs = oqw.run_gmm(
        oqw.constants.GMM.A_22,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        gmm_inputs,
        intensity_measure,
        periods=[period],
    )
    im_mean = next(column for column in gmm_outputs.columns if column.endswith("_mean"))
    im_std = next(
        column for column in gmm_outputs.columns if column.endswith("_std_Total")
    )
    output_ds = xr.Dataset(
        data_vars={
            "log_mean": (ground_motion_inputs.dims, gmm_outputs[im_mean].values),
            "log_stddev": (ground_motion_inputs.dims, gmm_outputs[im_std].values),
        },
        coords=ground_motion_inputs.coords,
    )
    return output_ds


@numba.njit(cache=True)
def _threshold_reduction(
    ground_motions: np.ndarray,
    n_sites: int,
    samples_per_rupture: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    # Initialize with the number of ruptures (len(samples))
    n_ruptures = len(samples_per_rupture)
    occupancy_matrix = np.zeros((n_sites, n_ruptures, len(thresholds)), dtype=np.int64)

    k = 0
    for i in range(n_sites):
        for j in range(n_ruptures):
            n_samples = samples_per_rupture[j]
            values = ground_motions[k : k + n_samples]
            threshold_vals = np.searchsorted(thresholds, values)
            for v in threshold_vals:
                occupancy_matrix[i, j, :v] += 1
            k += n_samples
    return occupancy_matrix


def monte_carlo_threshold_occupancy(
    ground_motion_observations: xr.Dataset,
    thresholds: npt.NDArray[np.floating],
    rng: np.random.Generator,
) -> xr.Dataset:
    """Compute rupture hazard for an array of thresholds with monte-carlo sampling.

    Parameters
    ----------
    ground_motions : pd.DataFrame
        Sampled ground motions.
    thresholds : npt.NDArray[np.floating]
        1D array of intensity measure thresholds.

    Returns
    -------
    npt.NDArray[np.floating]
        2D array of shape (n_sources, n_thresholds) containing the exceedance rates.
    """

    sampled_ground_motions = xr.apply_ufunc(
        rng.normal,
        ground_motion_observations.log_mean,
        ground_motion_observations.log_stddev,
    )
    sampled_ground_motions = sampled_ground_motions.sortby(["site", "rupture"])
    sites = np.unique(sampled_ground_motions.site.values)
    ruptures, sample_count = np.unique(
        sampled_ground_motions.rupture.values, return_counts=True
    )
    # sample count over counts by the number of sites.
    sample_count //= len(sites)
    values = sampled_ground_motions.values
    occupancy_count = _threshold_reduction(
        values,
        len(sites),
        sample_count,
        np.log(thresholds),
    )
    occupancy_count_da = xr.DataArray(
        occupancy_count,
        dims=("site", "rupture", "threshold"),
        coords=dict(site=sites, rupture=ruptures, threshold=thresholds),
    )
    sample_count_da = xr.DataArray(
        sample_count, dims="rupture", coords=dict(rupture=ruptures)
    )
    poe = occupancy_count_da / sample_count_da

    return poe


@numba.njit(parallel=True)
def _analytical_threshold_reduction(
    log_mean: np.ndarray, log_stddev: np.ndarray, z: np.ndarray, threshold: np.ndarray
) -> np.ndarray:

    (n_site, n_rupture, nz) = log_mean.shape
    nt = len(threshold)

    phi_z = truncnorm.pdf(z, float(STDDEV_LOWER), float(STDDEV_UPPER), 0.0, 1.0)

    weights = np.zeros(nz, dtype=np.float64)
    if nz > 1:
        weights[0] = 0.5 * (z[1] - z[0])
        weights[-1] = 0.5 * (z[-1] - z[-2])
        for m in range(1, nz - 1):
            weights[m] = 0.5 * (z[m + 1] - z[m - 1])
    elif nz == 1:
        weights[0] = 1.0

    phi_weights = phi_z * weights

    poe = np.zeros((n_site, n_rupture, nt), dtype=log_mean.dtype)
    sqrt2 = np.sqrt(2)

    for i in numba.prange(n_site):
        for j in range(n_rupture):
            for k in range(nt):
                t = threshold[k]
                integral_sum = 0.0

                for l in range(nz):
                    mu = log_mean[i, j, l]
                    sigma = log_stddev[i, j, l]

                    exceedance_prob = 0.5 * math.erfc((t - mu) / (sigma * sqrt2))

                    integral_sum += exceedance_prob * phi_weights[l]

                poe[i, j, k] = integral_sum

    return poe


def analytical_threshold_occupancy(
    ground_motion_observations: xr.Dataset, thresholds: npt.NDArray[np.floating]
) -> xr.DataArray:
    """Compute rupture hazard for an array of thresholds with z-score integration.

    Parameters
    ----------
    ground_motions : pd.DataFrame
        Sampled ground motions.
    thresholds : npt.NDArray[np.floating]
        1D array of intensity measure thresholds.

    Returns
    -------
    npt.NDArray[np.floating]
        2D array of shape (n_sources, n_thresholds) containing the exceedance rates.
    """
    cond_prob = _analytical_threshold_reduction(
        ground_motion_observations.log_mean.values,
        ground_motion_observations.log_stddev.values,
        ground_motion_observations.z.values,
        np.log(thresholds),
    )
    cond_prob_da = xr.DataArray(
        cond_prob,
        dims=("site", "rupture", "threshold"),
        coords=dict(
            site=ground_motion_observations.site.values,
            rupture=ground_motion_observations.rupture.values,
            threshold=thresholds,
        ),
    )
    return cond_prob_da


DEFAULT_PERIODS = [
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
]
THRESHOLDS = np.geomspace(0.1, 2.0, num=30)


def load_hazard_inputs(
    ruptures_path: Path, source_to_site_path: Path, sites_path: Path
):
    """Handles loading and basic indexing for input dataframes."""
    ruptures = pd.read_parquet(ruptures_path).rename_axis("rupture")
    source_to_site = pd.read_parquet(source_to_site_path).rename_axis("rupture")
    sites = gpd.read_parquet(sites_path).rename_axis("site")
    return ruptures, source_to_site, sites


def compute_distributed_hazard(
    path: Path, sites: gpd.GeoDataFrame, target_thresholds: np.ndarray
):
    """Processes distributed seismicity via nearest-neighbour spatial join."""
    ds_model = gpd.read_parquet(path).to_crs(sites.crs).query('rlz == "mean"')

    matched_sites = sites.sjoin_nearest(ds_model)
    ds_hazard = (
        matched_sites.reset_index()
        .set_index(["period", "threshold", "site"])["rate"]
        .to_xarray()
    )
    return ds_hazard.interp(threshold=target_thresholds)


def run_analytical_hazard(
    rates,
    realisations,
    rrup,
    sites,
    periods,
    thresholds,
):
    gmm_inputs = ground_motion_inputs(realisations, rrup, sites)
    gmm_hazards = []

    for period in tqdm.tqdm(periods, unit="period"):
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", period)
        gmm_outputs = gmm_outputs.unstack()
        cond_prob = analytical_threshold_occupancy(gmm_outputs, thresholds)
        hazard = cond_prob * rates.to_xarray()
        gmm_hazards.append(hazard)

    return xr.concat(gmm_hazards, dim=pd.Index(periods, name="period"))


def run_monte_carlo_simulation(
    realisations, rrup, sites, ruptures, n, column, periods, thresholds, seed
):
    """Core simulation loop for GMM hazard calculation."""
    gmm_inputs = ground_motion_inputs(realisations, rrup, sites)
    rng = np.random.default_rng(seed=seed)
    weights = (ruptures["rate"] / (ruptures[column] * n)).to_xarray()

    hazards = []
    for period in tqdm.tqdm(periods, unit="period"):
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", period)
        cond_prob = monte_carlo_threshold_occupancy(gmm_outputs, thresholds, rng)
        hazards.append(cond_prob * weights)

    return xr.concat(hazards, dim=pd.Index(periods, name="period"))


@app.command()
def hazard(
    ruptures_path: Path,
    source_to_site_path: Path,
    sites_path: Path,
    distributed_seismicity_path: Path,
    n: int,
    gmm_hazard_path: Path,
    periods: list[float] | None = None,
    thresholds: list[float] | None = None,
) -> None:
    ruptures, source_to_site, sites = load_hazard_inputs(
        ruptures_path, source_to_site_path, sites_path
    )
    periods = periods or DEFAULT_PERIODS

    realisations = analytical_rupture_sample(ruptures, STDDEV_LOWER, STDDEV_UPPER, n)
    rrup = rupture_distances(source_to_site).sel(site=list(sites.index.values))
    thresholds_arr = np.asarray(thresholds) if thresholds else THRESHOLDS
    rupture_hazard = run_analytical_hazard(
        ruptures["rate"], realisations, rrup, sites, periods, thresholds_arr
    )

    ds_hazard = compute_distributed_hazard(
        distributed_seismicity_path, sites, rupture_hazard.threshold.values
    )

    total_hazard = xr.Dataset(dict(ds_hazard=ds_hazard, hazard=rupture_hazard))
    total_hazard.to_netcdf(gmm_hazard_path, engine="h5netcdf")


@app.command()
def monte_carlo_hazard(
    ruptures_path: Path,
    source_to_site_path: Path,
    sites_path: Path,
    distributed_seismicity_path: Path,
    n: int,
    gmm_hazard_path: Path,
    periods: list[float] | None = None,
    thresholds: list[float] | None = None,
    seed: int | None = None,
    column: str = "kl_density",
) -> None:
    # 1. Setup Data
    ruptures, source_to_site, sites = load_hazard_inputs(
        ruptures_path, source_to_site_path, sites_path
    )
    periods = periods or DEFAULT_PERIODS
    thresholds_arr = np.asarray(thresholds) if thresholds else THRESHOLDS

    rrup = rupture_distances(source_to_site).sel(site=list(sites.index.values))

    realisations = monte_carlo_sample(ruptures, n, STDDEV_LOWER, STDDEV_UPPER, column)

    rupture_hazard = run_monte_carlo_simulation(
        realisations, rrup, sites, ruptures, n, column, periods, thresholds_arr, seed
    )

    ds_hazard = compute_distributed_hazard(
        distributed_seismicity_path, sites, rupture_hazard.threshold.values
    )

    total_hazard = xr.Dataset(dict(ds_hazard=ds_hazard, hazard=rupture_hazard))
    total_hazard.to_netcdf(gmm_hazard_path, engine="h5netcdf")


if __name__ == "__main__":
    app()
