"""Analytically integrate PSHA to obtain a hazard curve"""

import math
from pathlib import Path

import cyclopts
import geopandas as gpd
import numba
import numpy as np
import oq_wrapper as oqw
import pandas as pd
import scipy as sp
import tqdm
import xarray as xr
from numba_stats import norm

Array1 = np.ndarray[tuple[int,], np.dtype[np.float64]]
Array2 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Array3 = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
IArray1 = np.ndarray[tuple[int,], np.dtype[np.int64]]
IArray2 = np.ndarray[tuple[int, int], np.dtype[np.int64]]
IArray3 = np.ndarray[tuple[int, int, int], np.dtype[np.int64]]
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
    ruptures: pd.DataFrame,
    counts: pd.Series,
    z_lower: float,
    z_upper: float,
) -> xr.Dataset:
    sample = ruptures.loc[np.repeat(ruptures.index, counts)]
    rates = ruptures["rate"].loc[sample.index]
    mean_magnitude, sigma = get_leonard_magnitude_params(
        sample["area"] / 1e6, sample["rake"]
    )
    magnitudes = sp.stats.truncnorm.rvs(
        z_lower, z_upper, loc=mean_magnitude, scale=sigma
    )
    return xr.Dataset(
        dict(
            magnitudes=("rupture", magnitudes),
            rates=("rupture", rates),
        ),
        coords=dict(rupture=sample.index.values),
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
    return xr.Dataset(
        data_vars={
            "log_mean": (ground_motion_inputs.dims, gmm_outputs[im_mean].values),
            "log_stddev": (ground_motion_inputs.dims, gmm_outputs[im_std].values),
        },
        coords=ground_motion_inputs.coords,
    )


@numba.njit(parallel=True, fastmath=True, cache=True)
def _threshold_reduction_optimized(mean, stddev, samples, thresholds):
    n_sites, n_ruptures = mean.shape
    n_thresh = len(thresholds)
    occupancy_matrix = np.zeros((n_sites, n_ruptures, n_thresh), dtype=np.int64)

    for i in numba.prange(n_sites):
        for j in range(n_ruptures):
            n_samples = samples[j]
            if n_samples == 0:
                continue

            mu = mean[i, j]
            sigma = stddev[i, j]

            for _ in range(n_samples):
                val = np.random.normal(mu, sigma)
                idx = np.searchsorted(thresholds, val)
                for t in range(idx):
                    occupancy_matrix[i, j, t] += 1

    return occupancy_matrix


def monte_carlo_threshold_occupancy(
    ground_motion_observations: xr.Dataset,
    thresholds: Array1,
) -> xr.DataArray:

    thresholds_da = xr.DataArray(
        np.log(thresholds), dims=["threshold"], coords={"threshold": thresholds}
    )

    counts = xr.apply_ufunc(
        _threshold_reduction_optimized,
        ground_motion_observations.log_mean,  # Core dims: ['sites', 'ruptures']
        ground_motion_observations.log_stddev,  # Core dims: ['sites', 'ruptures']
        ground_motion_observations.samples,  # Core dims: ['ruptures']
        thresholds_da,  # Core dim: ['threshold']
        input_core_dims=[
            ["site", "rupture"],
            ["site", "rupture"],
            ["rupture"],
            ["threshold"],
        ],
        output_core_dims=[["site", "rupture", "threshold"]],
        vectorize=True,  # This handles the 'n_retries' or other extra dimensions
        dask="parallelized",
    )
    poe = counts / ground_motion_observations.samples
    return poe


@numba.njit(parallel=True, cache=True, fastmath=True)
def _analytical_threshold_reduction(
    log_mean: np.ndarray,
    log_stddev: np.ndarray,
    weights: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    (n_site, n_rupture, nz) = log_mean.shape
    nt = len(threshold)

    poe = np.zeros((n_site, n_rupture, nt), dtype=log_mean.dtype)

    for i in numba.prange(n_site):
        for j in range(n_rupture):
            for k in range(nt):
                t = threshold[k]
                integral_sum = 0.0
                for l in range(nz):
                    mu = log_mean[i, j, l]
                    sigma = log_stddev[i, j, l]
                    exceedance_prob = 1.0 - norm.cdf(np.array([t]), mu, sigma)[0]
                    integral_sum += exceedance_prob * weights[l]
                poe[i, j, k] = integral_sum

    return poe


def trapezium_integration_weights(
    z: Array1,
) -> Array1:
    phi_z = sp.stats.truncnorm.pdf(
        z, float(STDDEV_LOWER), float(STDDEV_UPPER), loc=0.0, scale=1.0
    )
    dz = z[1] - z[0]
    weights = phi_z * dz
    weights[[0, -1]] /= 2
    return weights


def analytical_threshold_occupancy(
    ground_motion_observations: xr.Dataset, thresholds: Array1, weights: xr.DataArray
) -> xr.DataArray:
    """
    Computes analytical threshold occupancy using survival functions and
    weighted integration over the z-dimension.
    """
    thresholds_da = xr.DataArray(
        np.log(thresholds), dims=["threshold"], coords={"threshold": thresholds}
    )

    exceedance_probs = xr.apply_ufunc(
        sp.stats.norm.sf,
        thresholds_da,
        ground_motion_observations.log_mean,
        ground_motion_observations.log_stddev,
        dask="allowed",
    )

    occupancy_prob = (exceedance_probs * weights).sum(dim="z")

    return occupancy_prob


def aggregate_analytical_hazard(
    gmm_outputs: xr.Dataset, rates: xr.DataArray, thresholds: Array1
) -> xr.DataArray:
    """Multiplies calculated occupancy probability by the rupture rate."""
    weights = trapezium_integration_weights(gmm_outputs.z.values)
    cond_prob = analytical_threshold_occupancy(gmm_outputs, thresholds, weights)
    return cond_prob * rates


def aggregate_monte_carlo_hazard(
    gmm_outputs: xr.Dataset,
    rates: xr.DataArray,
    thresholds: Array1,
    rng: np.random.Generator,
) -> xr.DataArray:
    """Multiplies simulated occupancy probability by calculated weights."""
    cond_prob = monte_carlo_threshold_occupancy(gmm_outputs, thresholds, rng)
    return cond_prob * rates


def calculate_analytical_hazard(
    ruptures: pd.DataFrame,
    source_to_site: pd.DataFrame,
    sites: pd.DataFrame,
    periods: Array1,
    thresholds: Array1,
    n: int,
) -> xr.DataArray:
    """End-to-end Python API for analytical hazard."""
    realisations = analytical_rupture_sample(ruptures, STDDEV_LOWER, STDDEV_UPPER, n)
    rrup = rupture_distances(source_to_site).sel(site=sites.index.values)
    gmm_inputs = ground_motion_inputs(realisations, rrup, sites)
    rates = ruptures["rate"].to_xarray()

    gmm_hazards = []
    for period in tqdm.tqdm(periods, unit="period"):
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", period).unstack()
        hazard = aggregate_analytical_hazard(gmm_outputs, rates, thresholds)
        gmm_hazards.append(hazard)

    return xr.concat(gmm_hazards, dim=pd.Index(periods, name="period"))


def calculate_monte_carlo_hazard(
    ruptures: pd.DataFrame,
    source_to_site: pd.DataFrame,
    sites: pd.DataFrame,
    periods: Array1,
    thresholds: Array1,
    n: int,
    column: str,
    seed: int | None,
) -> xr.DataArray:
    """End-to-end Python API for monte carlo hazard."""
    ruptures["count"] = np.round(n * ruptures[column]).astype(int)
    realisations = monte_carlo_sample(
        ruptures, ruptures["count"], STDDEV_LOWER, STDDEV_UPPER
    )
    rrup = rupture_distances(source_to_site).sel(site=sites.index.values)
    gmm_inputs = ground_motion_inputs(realisations.magnitudes, rrup, sites)

    rng = np.random.default_rng(seed=seed)

    hazards = []
    pbar = tqdm.tqdm(periods, unit="Period", position=1, leave=False)
    for period in pbar:
        pbar.set_description(f"pSA({period:.2f})")
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", period)
        hazard = aggregate_monte_carlo_hazard(
            gmm_outputs,
            realisations.rates.drop_duplicates("rupture"),
            thresholds,
            rng,
        )
        hazards.append(hazard)

    return xr.concat(hazards, dim=pd.Index(periods, name="period"))


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
    ruptures = pd.read_parquet(ruptures_path).rename_axis("rupture")
    source_to_site = pd.read_parquet(source_to_site_path).rename_axis("rupture")
    sites = gpd.read_parquet(sites_path).rename_axis("site")
    return ruptures, source_to_site, sites


def compute_distributed_hazard(
    path: Path, sites: gpd.GeoDataFrame, target_thresholds: np.ndarray
):
    ds_model = gpd.read_parquet(path).to_crs(sites.crs).query('rlz == "mean"')
    matched_sites = sites.sjoin_nearest(ds_model)
    ds_hazard = (
        matched_sites.reset_index()
        .set_index(["period", "threshold", "site"])["rate"]
        .to_xarray()
    )
    return ds_hazard.interp(threshold=target_thresholds)


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
    periods_arr = np.array(periods or DEFAULT_PERIODS)
    thresholds_arr = np.asarray(thresholds) if thresholds else THRESHOLDS

    rupture_hazard = calculate_analytical_hazard(
        ruptures, source_to_site, sites, periods_arr, thresholds_arr, n
    )

    ds_hazard = compute_distributed_hazard(
        distributed_seismicity_path, sites, rupture_hazard.threshold.values
    )
    xr.Dataset(dict(ds_hazard=ds_hazard, hazard=rupture_hazard)).to_netcdf(
        gmm_hazard_path, engine="h5netcdf"
    )


@app.command()
def monte_carlo_hazard(
    ruptures_path: Path,
    source_to_site_path: Path,
    sites_path: Path,
    distributed_seismicity_path: Path,
    n: int,
    gmm_hazard_path: Path,
    num_realisations: int = 10,  # Added this
    periods: list[float] | None = None,
    thresholds: list[float] | None = None,
    seed: int | None = None,
    column: str = "kl_density",
) -> None:
    ruptures, source_to_site, sites = load_hazard_inputs(
        ruptures_path, source_to_site_path, sites_path
    )
    periods_arr = np.array(periods or DEFAULT_PERIODS)
    thresholds_arr = np.asarray(thresholds) if thresholds else THRESHOLDS

    # Collect ensemble results
    all_hazard_results = []

    # Range is from seed to seed + num_realisations to ensure independence
    pbar_outer = tqdm.trange(num_realisations, desc="Realisations", position=0)
    for i in pbar_outer:
        current_seed = (seed + i) if seed is not None else None
        pbar_outer.set_description(
            f"Realisation {i + 1}/{num_realisations} (Seed: {current_seed})"
        )
        run_hazard = calculate_monte_carlo_hazard(
            ruptures,
            source_to_site,
            sites,
            periods_arr,
            thresholds_arr,
            n,
            column,
            current_seed,
        )
        all_hazard_results.append(run_hazard.sum("rupture"))

    # Concatenate along a new dimension: 'realisation'
    rupture_hazard_ensemble = xr.concat(
        all_hazard_results, dim=pd.Index(range(num_realisations), name="realisation")
    )

    # Compute distributed hazard once (assuming it's deterministic/background)
    ds_hazard = compute_distributed_hazard(
        distributed_seismicity_path, sites, rupture_hazard_ensemble.threshold.values
    )

    # Save the ensemble dataset
    xr.Dataset(dict(ds_hazard=ds_hazard, hazard=rupture_hazard_ensemble)).to_netcdf(
        gmm_hazard_path, engine="h5netcdf"
    )


if __name__ == "__main__":
    app()
