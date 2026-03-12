"""Analytically integrate PSHA to obtain a hazard curve"""

import functools
import math
from pathlib import Path

import cyclopts
import geopandas as gpd
import numba
import numpy as np
import oq_wrapper as oqw
import pandas as pd
import scipy as sp
import shapely
import tqdm
import tqdm.contrib.concurrent
import xarray as xr

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
            mag=("realisation", magnitudes),
            rates=("realisation", rates),
        ),
        coords=dict(
            realisation=np.arange(len(sample)),
            rupture=("realisation", sample.index.values),
        ),
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
    return rupture_parameters.assign(
        vs30=vs30,
        rrup=rupture_distances.sel(
            rupture=rupture_parameters.rupture, site=sites.index
        ),
    )


RRUP_INTERPOLANTS = np.array(
    [
        [
            3.5,
            3.6,
            3.7,
            3.8,
            3.9,
            4.0,
            4.1,
            4.2,
            4.3,
            4.4,
            4.5,
            4.6,
            4.7,
            4.8,
            4.9,
            5.0,
            5.1,
            5.2,
            5.3,
            5.4,
            5.5,
            5.6,
            5.7,
            5.8,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
        ],
        [
            96.001584,
            95.96318337,
            98.0,
            102.0,
            108.02690269,
            114.86856676,
            123.44523863,
            128.68959965,
            134.58683383,
            145.68111725,
            157.68992642,
            170.68864766,
            188.45405921,
            192.22314039,
            203.98873437,
            216.47447683,
            233.19667576,
            248.66112894,
            258.19003824,
            268.72840715,
            280.03281854,
            297.1730673,
            300.0,
            300.0,
            300.0,
            300.0,
            300.0,
            300.0,
            300.0,
        ],
    ]
)


def estimate_compute(
    ruptures: pd.DataFrame, clip_geometry: shapely.Geometry, resolution: float
) -> pd.DataFrame:
    mean_mag, _ = get_leonard_magnitude_params(ruptures["area"] / 1e6, ruptures["rake"])

    rrup = pd.Series(
        np.interp(mean_mag, RRUP_INTERPOLANTS[0], RRUP_INTERPOLANTS[1]),
        index=mean_mag.index,
    )
    fault_area = shapely.intersection(
        shapely.buffer(ruptures["geometry"], rrup), clip_geometry
    )

    bounding_box = shapely.minimum_rotated_rectangle(fault_area)
    corner_distance = np.sqrt(2) * rrup
    ctx = pd.DataFrame(dict(rrup=corner_distance))
    ctx["mag"] = mean_mag
    ctx["rake"] = ruptures["rake"]
    vs30 = 500.0
    ctx["vs30"] = vs30
    ctx["z1pt0"] = oqw.estimations.chiou_young_08_calc_z1p0(vs30)

    ds595_output = oqw.run_gmm(
        oqw.constants.GMM.AS_16, oqw.constants.TectType.ACTIVE_SHALLOW, ctx, "Ds595"
    )
    ds595 = np.exp(ds595_output["Ds595_mean"])
    dt = resolution / 20.0
    nt = (corner_distance / 3.5 + ds595) / dt
    area = shapely.area(bounding_box) / (1e6 * resolution)
    compute = area * nt
    compute_df = pd.DataFrame(dict(compute=compute))
    compute_df["nt"] = nt
    compute_df["area"] = area
    return compute_df


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


@numba.njit(parallel=True, cache=True)
def _threshold_reduction_optimized(mean, stddev, thresholds):
    n_sites, n_realisations = mean.shape
    n_thresh = len(thresholds)
    occupancy_matrix = np.zeros((n_sites, n_realisations, n_thresh), dtype=np.int64)

    for i in numba.prange(n_sites):
        for j in range(n_realisations):
            mu = mean[i, j]
            sigma = stddev[i, j]

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
        thresholds_da,  # Core dim: ['threshold']
        input_core_dims=[
            ["site", "realisation"],
            ["site", "realisation"],
            ["threshold"],
        ],
        output_core_dims=[["site", "realisation", "threshold"]],
        vectorize=True,  # This handles the 'n_retries' or other extra dimensions
        dask="parallelized",
    )

    poe = counts.groupby(["site", "rupture"]).mean()
    return poe


@numba.guvectorize(
    ["(float64, float64, float64, float64[:])"],
    "(),(),()->()",
    nopython=True,
    # Empirically, fastmath is ok here it barely changes the
    # difference in hazard. Disabled anyway because it only shaves a
    # minute off total hazard calculations.
    target="parallel",
)
def fast_norm_sf_numba(x, mu, sigma, res):
    z = (x - mu) / sigma
    # As opposed to 1.0 + math.erf because of accuracy issues.
    res[0] = 0.5 * math.erfc(z / np.sqrt(2.0))


def analytical_threshold_occupancy(
    ground_motion_observations: xr.Dataset,
    thresholds: Array1,
) -> xr.DataArray:
    """
    Computes analytical threshold occupancy using survival functions and
    weighted integration over the z-dimension.
    """
    thresholds_da = xr.DataArray(
        np.log(thresholds), dims=["threshold"], coords={"threshold": thresholds}
    )
    exceedance_probs = xr.apply_ufunc(
        fast_norm_sf_numba,
        thresholds_da,
        ground_motion_observations.log_mean,
        ground_motion_observations.log_stddev,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        dask="forbidden",
    )

    phi_z = xr.apply_ufunc(
        functools.partial(
            sp.stats.truncnorm.pdf,
            a=float(STDDEV_LOWER),
            b=float(STDDEV_UPPER),
            loc=0.0,
            scale=1.0,
        ),
        ground_motion_observations.z,
    )

    exceedance_probs *= phi_z
    occupancy_prob = exceedance_probs.integrate("z")

    return occupancy_prob


def aggregate_analytical_hazard(
    gmm_outputs: xr.Dataset, rates: xr.DataArray, thresholds: Array1
) -> xr.DataArray:
    """Multiplies calculated occupancy probability by the rupture rate."""
    cond_prob = analytical_threshold_occupancy(gmm_outputs, thresholds)
    cond_prob *= rates
    return cond_prob


def aggregate_monte_carlo_hazard(
    gmm_outputs: xr.Dataset,
    rates: xr.DataArray,
    thresholds: Array1,
) -> xr.DataArray:
    """Multiplies simulated occupancy probability by calculated weights."""
    cond_prob = monte_carlo_threshold_occupancy(gmm_outputs, thresholds)
    cond_prob *= rates.sel(rupture=cond_prob.rupture)
    return cond_prob


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
    count = np.round(n * ruptures[column] / ruptures[column].sum()).astype(int)
    realisations = monte_carlo_sample(ruptures, count, STDDEV_LOWER, STDDEV_UPPER)
    rrup = rupture_distances(source_to_site).sel(site=sites.index.values)
    gmm_inputs = ground_motion_inputs(realisations, rrup, sites)
    np.random.seed(seed=seed)
    hazards = []

    for period in periods:
        gmm_outputs = run_ground_motion_model(
            gmm_inputs.stack(sample=gmm_inputs.dims), "pSA", period
        ).unstack()
        hazard = aggregate_monte_carlo_hazard(
            gmm_outputs,
            ruptures["rate"].to_xarray(),
            thresholds,
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


def _run_single_realisation(
    i, seed, ruptures, source_to_site, sites, periods_arr, thresholds_arr, n, column
):
    """Worker function for a single Monte Carlo realization."""
    current_seed = (seed + i) if seed is not None else None
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
    return run_hazard.sum("rupture")


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
    num_realisations: int = 10,
    periods: list[float] | None = None,
    thresholds: list[float] | None = None,
    seed: int | None = None,
    column: str = "kl_density",
    max_workers: int = 1,
) -> None:
    ruptures, source_to_site, sites = load_hazard_inputs(
        ruptures_path, source_to_site_path, sites_path
    )
    periods_arr = np.array(periods or DEFAULT_PERIODS)
    thresholds_arr = np.asarray(thresholds) if thresholds else THRESHOLDS

    worker_func = functools.partial(
        _run_single_realisation,
        seed=seed,
        ruptures=ruptures,
        source_to_site=source_to_site,
        sites=sites,
        periods_arr=periods_arr,
        thresholds_arr=thresholds_arr,
        n=n,
        column=column,
    )

    all_hazard_results = tqdm.contrib.concurrent.process_map(
        worker_func,
        range(num_realisations),
        max_workers=max_workers,
        chunksize=1,
        desc="Realisations",
    )

    rupture_hazard_ensemble = xr.concat(
        all_hazard_results, dim=pd.Index(range(num_realisations), name="realisation")
    )

    ds_hazard = compute_distributed_hazard(
        distributed_seismicity_path, sites, rupture_hazard_ensemble.threshold.values
    )

    xr.Dataset(dict(ds_hazard=ds_hazard, hazard=rupture_hazard_ensemble)).to_netcdf(
        gmm_hazard_path, engine="h5netcdf"
    )


if __name__ == "__main__":
    app()
