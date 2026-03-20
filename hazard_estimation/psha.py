"""Analytically integrate PSHA to obtain a hazard curve"""

import functools
import math
from pathlib import Path

import cyclopts
import dask.array as da
import flox.xarray
import geopandas as gpd
import numba
import numpy as np
import oq_wrapper as oqw
import pandas as pd
import scipy as sp
import shapely
import tqdm
import xarray as xr
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

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
) -> xr.Dataset:
    mean_magnitude, sigma = get_leonard_magnitude_params(
        ruptures["area"].values / 1e6, ruptures["rake"]
    )
    z = np.linspace(z_lower, z_upper, num=num)
    magnitude = mean_magnitude[:, None] + z[None, :] * sigma[:, None]

    magnitude_da = xr.DataArray(
        magnitude,
        dims=("rupture", "z"),
        coords=dict(z=z, rupture=ruptures.index.values),
        name="mag",
    )
    return xr.merge(
        [ruptures.drop(columns=["mag", "geometry"]).to_xarray(), magnitude_da]
    )


def rupture_distances(source_to_site: pd.DataFrame) -> xr.Dataset:
    return source_to_site.reset_index().set_index(["site", "rupture"]).to_xarray()


def ground_motion_inputs(
    rupture_parameters: xr.DataArray,
    rupture_distances: xr.Dataset,
    sites: pd.DataFrame,
) -> xr.Dataset:
    sites_arr = (
        sites[["vs30", "Z1.0", "Z2.5"]]
        .to_xarray()
        .rename({"Z1.0": "z1pt0", "Z2.5": "z2pt5"})
    )

    return xr.merge([rupture_parameters, rupture_distances, sites_arr], join="inner")


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


def gmm_worker(
    ds_chunk: xr.Dataset, intensity_measure: str, period: float
) -> xr.Dataset:
    """
    This is your original logic, acting on a single chunk of data.
    """
    # 1. Logic: Stack, Convert to DF, Add Metadata
    stacked = ds_chunk.stack(sample=ds_chunk.dims)
    df = stacked.to_dataframe()

    df["vs30measured"] = True
    df["ztor"] = 0.0
    df["hypo_depth"] = df["zbot"] / 2.0

    # 2. Run GMM
    gmm_outputs = oqw.run_gmm_logic_tree(
        oqw.constants.GMMLogicTree.NSHM2022,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        df,
        intensity_measure,
        periods=[period],
    )

    # 3. Convert back to xarray and identify columns
    # We use the index from the original 'stacked' to ensure alignment
    gmm_ds = gmm_outputs.to_xarray()

    variables = list(gmm_ds.data_vars)
    im_mean = next(c for c in variables if c.endswith("_mean"))
    im_std = next(c for c in variables if c.endswith("_std_Total"))

    # 4. Final selection and unstacking back to original dims
    # We rename and ensure the dimensions match the template (z, site, rupture)
    res = gmm_ds[[im_mean, im_std]].rename({im_mean: "log_mean", im_std: "log_stddev"})

    return res.transpose("z", "site", "rupture")


def run_ground_motion_model(
    ds: xr.Dataset, intensity_measure: str, period: float
) -> xr.Dataset:
    chunked = ds.chunk({"z": -1, "site": -1, "rupture": 100})

    # 2. Define the Template
    shape = (chunked.sizes["z"], chunked.sizes["site"], chunked.sizes["rupture"])

    # FIX: Convert the xarray chunksizes dict to a positional tuple
    # This ensures Dask gets ( (15,), (212,), (500, 500, ...) )
    dask_chunks = tuple(chunked.chunksizes[d] for d in ("z", "site", "rupture"))

    template = xr.Dataset(
        data_vars={
            "log_mean": (
                ("z", "site", "rupture"),
                da.empty(shape, chunks=dask_chunks, dtype=float),
            ),
            "log_stddev": (
                ("z", "site", "rupture"),
                da.empty(shape, chunks=dask_chunks, dtype=float),
            ),
        },
        coords=chunked.coords,
    )

    # 3. Parallel Execution
    return chunked.map_blocks(
        gmm_worker,
        kwargs={"intensity_measure": intensity_measure, "period": period},
        template=template,
    ).compute()


@numba.guvectorize(
    [(numba.float64, numba.float64, numba.float64[:], numba.int8[:])],
    "(),(),(t)->(t)",
    cache=True,
)
def _fast_threshold_mask(mu, sigma, thresholds, out):
    val = np.random.normal(mu, sigma)

    for t in range(thresholds.shape[0]):
        if val > thresholds[t]:
            out[t] = 1
        else:
            out[t] = 0


def monte_carlo_threshold_occupancy(
    ground_motion_observations: xr.Dataset,
    thresholds: Array1,
) -> xr.DataArray:

    thresholds_da = xr.DataArray(
        np.log(thresholds), dims=["threshold"], coords={"threshold": thresholds}
    )

    counts = xr.apply_ufunc(
        _fast_threshold_mask,
        ground_motion_observations.log_mean,
        ground_motion_observations.log_stddev,
        thresholds_da,
        input_core_dims=[[], [], ["threshold"]],
        output_core_dims=[["threshold"]],
        output_dtypes=[np.int8],
    )
    expected_sites = ground_motion_observations.site.values
    expected_ruptures = np.unique(ground_motion_observations.rupture.values)
    poe = flox.xarray.xarray_reduce(
        counts,
        "site",
        "rupture",
        func="mean",
        method="blockwise",
        expected_groups=(
            expected_sites,
            expected_ruptures,
        ),
    )

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
    source_to_site: xr.Dataset,
    sites: pd.DataFrame,
    periods: Array1,
    thresholds: Array1,
    n: int,
) -> xr.DataArray:
    """End-to-end Python API for analytical hazard."""
    realisations = analytical_rupture_sample(ruptures, STDDEV_LOWER, STDDEV_UPPER, n)

    gmm_inputs = ground_motion_inputs(realisations, source_to_site, sites)
    rates = ruptures["rate"].to_xarray()

    gmm_hazards = []
    for period in tqdm.tqdm(periods, unit="period"):
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", period)
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
    pbar = tqdm.tqdm(periods, unit="period", position=1, leave=False)

    for period in pbar:
        pbar.set_description(f"pSA({period:.2f})")
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", period)
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
    source_to_site = xr.open_dataset(source_to_site_path, engine="h5netcdf")
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
    # 1. Define what a single SLURM job (one "worker") looks like
    cluster = SLURMCluster(
        cores=8,  # CPUs per SLURM job
        memory="32GB",  # RAM per SLURM job
        walltime="01:00:00",  # How long each worker should live
    )

    # 2. Tell SLURM to actually start the workers
    # This sends out 10 sbatch jobs. Total: 80 cores!
    cluster.scale(jobs=10)

    client = Client(cluster)
    app()
