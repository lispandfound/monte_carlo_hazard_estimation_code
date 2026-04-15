"""Analytically integrate PSHA to obtain a hazard curve"""

import parse

from distributed import Client, LocalCluster

import numpy.typing as npt
import psutil
import functools
import math
from pathlib import Path

from dask.diagnostics import ProgressBar
import dask.array as da
import cyclopts
import geopandas as gpd
import numba
import numpy as np
import oq_wrapper as oqw
import pandas as pd
import scipy as sp
import shapely
import tqdm
import xarray as xr
from openquake.hazardlib.cross_correlation import BakerJayaram2008
from openquake.hazardlib.imt import SA

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


def singleton_array(arr: npt.ArrayLike | list, name: str) -> xr.DataArray:
    return xr.DataArray(arr, dims=[name], coords={name: arr}, name=name)


def get_leonard_magnitude_params(
    area: xr.DataArray, rake: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
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
    mu_constant = xr.where(is_strike_slip, 3.99, 4.03)
    sigma = xr.where(is_strike_slip, 0.26, 0.30).rename("sigma")
    mean_magnitude = np.log10(area) + mu_constant
    mean_magnitude = mean_magnitude.rename("mean_magnitude")
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
    sample["mag"] = magnitudes
    dset = (
        sample[["rake", "area", "mag", "zbot", "dip"]]
        .reset_index()
        .rename_axis("sample")
        .to_xarray()
    )
    dset = dset.set_coords("rupture")
    return dset


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
    rupture_parameters: xr.Dataset,
    rupture_distances: xr.Dataset,
    sites: xr.Dataset,
) -> xr.Dataset:
    sites_arr = sites[["vs30", "Z1.0", "Z2.5"]].rename(
        {"Z1.0": "z1pt0", "Z2.5": "z2pt5"}
    )
    potential_inputs = {
        "mag",
        "mean_mag",
        "mag_sigma",
        "area",
        "rake",
        "rate",
        "dip",
        "zbot",
        "rrup",
        "rx",
        "ry",
        "rjb",
        "vs30",
        "z1pt0",
        "z2pt5",
    }

    dset = xr.merge(
        [rupture_parameters, rupture_distances, sites_arr],
        join="inner",
        compat="no_conflicts",
    )

    dset["mean_mag"] = mean_magnitude
    dset["mag_sigma"] = sigma
    dset = dset.drop_vars("mag")

    variables = set(dset)
    gmm_variables = potential_inputs & variables
    return dset[gmm_variables]


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
    ruptures: xr.Dataset,
    source_to_site: xr.Dataset,
    sites: xr.Dataset,
    periods: xr.DataArray,
    models: xr.DataArray,
    intensity_measure: str,
) -> xr.Dataset:
    inputs = xr.merge([ruptures, source_to_site, sites], join="inner")

    original_dims = list(inputs.dims)
    stacked = inputs.stack(coord=original_dims)

    df = stacked.to_dataframe()
    df["vs30measured"] = True
    df["ztor"] = 0.0
    df["hypo_depth"] = df["zbot"] / 2.0

    model_results = []

    # Patterns for parsing columns
    mean_pattern = "pSA_{p:f}_mean"
    std_pattern = "pSA_{p:f}_std_Total"

    for model in models.values:
        gmm_outputs = oqw.run_gmm(
            model,
            oqw.constants.TectType.ACTIVE_SHALLOW,
            df,
            intensity_measure,
            periods=periods.values,
        )

        period_data = {p: {"mean": None, "std": None} for p in periods.values}

        for column in gmm_outputs.columns:
            if res := parse.parse(mean_pattern, column):
                p_val = res["p"]
                period_data[p_val]["mean"] = gmm_outputs[column].values
            elif res := parse.parse(std_pattern, column):
                p_val = res["p"]
                period_data[p_val]["std"] = gmm_outputs[column].values

        means_stacked = np.stack(
            [period_data[p]["mean"] for p in periods.values], axis=1
        )
        stds_stacked = np.stack([period_data[p]["std"] for p in periods.values], axis=1)

        ds_model = xr.Dataset(
            data_vars={
                "log_mean": (["coord", "period"], means_stacked.astype(np.float32)),
                "log_stddev": (["coord", "period"], stds_stacked.astype(np.float32)),
            },
            coords={
                "coord": stacked.coord,
                "period": periods.values,
            },
        )
        model_results.append(ds_model)

    da_all_models = xr.concat(
        model_results, dim=xr.DataArray(models.values, name="gmm", dims=["gmm"])
    )

    final_output = da_all_models.unstack("coord")

    return final_output.transpose("gmm", "period", "rupture", "site", "z")


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


@numba.guvectorize(
    ["(float64, float64, float64, float64[:])"],
    "(),(),()->()",
    nopython=True,
    # Empirically, fastmath is ok here it barely changes the
    # difference in hazard. Disabled anyway because it only shaves a
    # minute off total hazard calculations.
    target="cpu",
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
    logic_tree: bool,
) -> xr.DataArray:
    """End-to-end Python API for analytical hazard."""
    realisations = analytical_rupture_sample(ruptures, STDDEV_LOWER, STDDEV_UPPER, n)

    gmm_inputs = ground_motion_inputs(realisations, source_to_site, sites)
    rates = ruptures["rate"].to_xarray()

    gmm_hazards = []
    for period in tqdm.tqdm(periods, unit="period"):
        gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", logic_tree)
        hazard = aggregate_analytical_hazard(gmm_outputs, rates, thresholds)
        gmm_hazards.append(hazard)

    return xr.concat(gmm_hazards, dim=pd.Index(periods, name="period"))


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
    df_ruptures = pd.read_parquet(ruptures_path)[["mag", "area", "rake", "zbot", "dip"]]
    rupture_ids = df_ruptures.index.values
    ruptures = xr.Dataset.from_dataframe(df_ruptures)
    ruptures = ruptures.assign_coords(rupture=rupture_ids)

    source_to_site = xr.open_dataset(source_to_site_path, engine="h5netcdf")

    gdf_sites = gpd.read_parquet(sites_path)
    site_ids = gdf_sites.index.values.astype(str)  # Force to string array

    sites_subset = gdf_sites[["vs30", "Z1.0", "Z2.5"]].rename(
        columns={"Z1.0": "z1pt0", "Z2.5": "z2pt5"}
    )

    sites = xr.Dataset.from_dataframe(pd.DataFrame(sites_subset))
    sites = sites.assign_coords(site=site_ids)

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
    logic_tree: bool = False,
) -> None:
    ruptures, source_to_site, sites = load_hazard_inputs(
        ruptures_path, source_to_site_path, sites_path
    )
    periods_arr = np.array(periods or DEFAULT_PERIODS)
    thresholds_arr = np.asarray(thresholds) if thresholds else THRESHOLDS

    rupture_hazard = calculate_analytical_hazard(
        ruptures, source_to_site, sites, periods_arr, thresholds_arr, n, logic_tree
    )

    ds_hazard = compute_distributed_hazard(
        distributed_seismicity_path, sites, rupture_hazard.threshold.values
    )
    xr.Dataset(dict(ds_hazard=ds_hazard, hazard=rupture_hazard)).to_netcdf(
        gmm_hazard_path, engine="h5netcdf"
    )


def draw_ruptures(
    density: xr.DataArray,
    num_samples: int,
    random: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if random:
        counts = rng.multinomial(num_samples, density)
    else:
        norm_density = density / density.sum()

        def objective(effective_sample_count: int) -> int:
            return np.round(effective_sample_count * norm_density).sum() - num_samples

        effective_sample_count = sp.optimize.bisect(objective, 0, 2 * num_samples)
        counts = np.round(effective_sample_count * density / density.sum()).astype(int)

    return np.repeat(density.rupture.values, counts)


def bin_magnitudes(
    mean_magnitude: xr.DataArray, sigma: xr.DataArray, num_bins: int
) -> xr.DataArray:
    z = np.linspace(-1, 1, num=num_bins)
    z_arr = singleton_array(z, "z")
    return mean_magnitude + z_arr * sigma


def logic_tree_with_weights() -> xr.DataArray:
    epistemic_branch_lower = -1.2815
    epistemic_branch_upper = 1.2815
    epistemic_branch_central = 0
    logic_tree_data = [
        (oqw.constants.GMM.S_22, epistemic_branch_upper, 0.117),
        (oqw.constants.GMM.S_22, epistemic_branch_central, 0.156),
        (oqw.constants.GMM.S_22, epistemic_branch_lower, 0.117),
        (oqw.constants.GMM.A_22, epistemic_branch_upper, 0.084),
        (oqw.constants.GMM.A_22, epistemic_branch_central, 0.112),
        (oqw.constants.GMM.A_22, epistemic_branch_lower, 0.084),
        (oqw.constants.GMM.ASK_14, epistemic_branch_upper, 0.0198),
        (oqw.constants.GMM.ASK_14, epistemic_branch_central, 0.0264),
        (oqw.constants.GMM.ASK_14, epistemic_branch_lower, 0.0198),
        (oqw.constants.GMM.BSSA_14, epistemic_branch_upper, 0.0198),
        (oqw.constants.GMM.BSSA_14, epistemic_branch_central, 0.0264),
        (oqw.constants.GMM.BSSA_14, epistemic_branch_lower, 0.0198),
        (oqw.constants.GMM.CB_14, epistemic_branch_upper, 0.0198),
        (oqw.constants.GMM.CB_14, epistemic_branch_central, 0.0264),
        (oqw.constants.GMM.CB_14, epistemic_branch_lower, 0.0198),
        (oqw.constants.GMM.CY_14, epistemic_branch_upper, 0.0198),
        (oqw.constants.GMM.CY_14, epistemic_branch_central, 0.0264),
        (oqw.constants.GMM.CY_14, epistemic_branch_lower, 0.0198),
        (oqw.constants.GMM.Br_13, epistemic_branch_upper, 0.0198),
        (oqw.constants.GMM.Br_13, epistemic_branch_central, 0.0264),
        (oqw.constants.GMM.Br_13, epistemic_branch_lower, 0.0198),
    ]
    df = pd.DataFrame(
        logic_tree_data, columns=["gmm", "epistemic_branch", "weight"]
    ).set_index(["gmm", "epistemic_branch"])
    weights = df["weight"]
    da = weights.to_xarray()
    da.name = "weights"
    return da


def map_blocks_template_for(ruptures, sites, periods, gmms) -> xr.Dataset:
    z = ruptures.z.values
    ruptures = ruptures.rupture.values
    sites = sites.site.values
    gmms = gmms.gmm.values

    array = da.empty(
        (len(gmms), len(periods), len(ruptures), len(sites), len(z)), dtype=np.float32
    )
    return xr.Dataset(
        dict(
            log_mean=(("gmm", "period", "rupture", "site", "z"), array),
            log_stddev=(("gmm", "period", "rupture", "site", "z"), array),
        ),
        coords=dict(rupture=ruptures, site=sites, period=periods, gmm=gmms, z=z),
    )


@app.command()
def ground_motion_database(
    ruptures_path: Path,
    source_to_site_path: Path,
    sites_path: Path,
    output: Path,
    periods: list[float] | None = None,
    logic_tree: bool = False,
    rupture_chunk: int = 50,
    site_chunk: int = 50,
    z_chunk: int = -1,
) -> None:
    ruptures, source_to_site, sites = load_hazard_inputs(
        ruptures_path, source_to_site_path, sites_path
    )
    source_to_site = source_to_site.sel(
        rupture=ruptures.rupture.values, site=sites.site.values
    )

    mean_magnitude, sigma = get_leonard_magnitude_params(
        ruptures["area"] / 1e6, ruptures["rake"]
    )
    magnitudes = bin_magnitudes(mean_magnitude, sigma, num_bins=15)
    ruptures["mag"] = magnitudes

    periods_da = singleton_array(np.array(periods or DEFAULT_PERIODS), "period")

    models = logic_tree_with_weights()
    chunks = dict(rupture=rupture_chunk, z=z_chunk, site=site_chunk, gmm=-1, period=-1)

    template = map_blocks_template_for(ruptures, sites, periods_da, models)
    template = template.chunk(chunks)
    source_to_site = source_to_site.chunk(dict(rupture=rupture_chunk))
    with (
        LocalCluster(host="0.0.0.0", dashboard_address=":8787") as cluster,
        Client(cluster) as client,
    ):
        print(f"Dashboard at {client.dashboard_link}")

        gmm_outputs = xr.map_blocks(
            run_ground_motion_model,
            ruptures.chunk(dict(rupture=rupture_chunk)),
            args=[
                source_to_site.chunk(dict(rupture=rupture_chunk, site=site_chunk)),
                sites.chunk(dict(site=site_chunk)),
            ],
            kwargs=dict(intensity_measure="pSA", periods=periods_da, models=models.gmm),
            template=template,
        )
        with ProgressBar():
            gmm_outputs.to_zarr(output, mode="w")
    # gmm_outputs = run_ground_motion_model(
    #     gmm_inputs, magnitude_variate, "pSA", logic_tree
    # )
    # gmm_outputs = run_ground_motion_model(gmm_inputs, "pSA", logic_tree)


def draw_z_scores(z: xr.DataArray, n: int, rng: np.random.Generator) -> np.ndarray:
    dist = sp.stats.truncnorm(STDDEV_LOWER, STDDEV_UPPER)
    z_arr = z.values
    dz = z_arr[1] - z_arr[0]
    if not np.all(np.isclose(np.diff(z_arr), dz)):
        raise ValueError("Z-score sampling assumes dz is constant")
    start = z_arr - dz / 2
    end = z_arr + dz / 2
    weights = (dist.cdf(end) - dist.cdf(start)).astype(np.float32)

    z_values = rng.choice(
        z_arr,
        size=n,
        p=weights,
        replace=True,
    )
    return z_values.astype(np.float32)


def draw_models(
    logic_tree: xr.DataArray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    logic_tree_df = logic_tree.to_dataframe().reset_index()

    sample = logic_tree_df.sample(n, replace=True, random_state=rng, weights="weights")

    return sample["gmm"].values, sample["epistemic_branch"].values.astype(np.float32)


def create_sample_array(
    ruptures: np.ndarray, z: np.ndarray, models: np.ndarray, branches: np.ndarray
) -> xr.Dataset:
    return xr.Dataset(
        dict(
            rupture=("sample", ruptures),
            z=("sample", z),
            model=("sample", models),
            branch=("sample", branches),
        ),
        coords=dict(sample=np.arange(len(z))),
    )


def sample_ground_motions(sample_array: xr.Dataset, gmm_database: xr.Dataset):
    selection = dict(
        rupture=sample_array["rupture"],
        z=sample_array["z"],
        gmm=sample_array["model"],
        site=sample_array["site"],
    )
    ground_motions = sample_array["branch"].copy()
    ground_motions += sample_array["epsilon"]
    ground_motions *= gmm_database["log_stddev"].sel(selection)
    ground_motions += gmm_database["log_mean"].sel(selection)

    return ground_motions


def draw_random_variate(
    like: np.ndarray, correlation: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    z = rng.normal(size=like.shape)
    l = np.linalg.cholesky(correlation)
    return (l @ z[..., np.newaxis]).squeeze().astype(np.float32)


def period_correlation_matrix(periods: np.ndarray) -> np.ndarray:
    model = BakerJayaram2008()
    return model.get_cross_correlation_mtx([SA(p) for p in periods])


def draw_sample_inputs(
    n: int,
    rupture_density: xr.DataArray,
    logic_tree: xr.DataArray,
    gmm_database: xr.Dataset,
    rng: np.random.Generator,
) -> xr.Dataset:
    sample_ruptures = draw_ruptures(
        rupture_density,
        n,
        rng=rng,
        random=False,
    )
    sample_z_scores = draw_z_scores(gmm_database["z"], n, rng=rng)
    logic_tree = logic_tree_with_weights()
    sample_models, sample_branches = draw_models(logic_tree, n, rng=rng)
    sample_array = create_sample_array(
        sample_ruptures, sample_z_scores, sample_models, sample_branches
    )
    # Introduce sites and periods as additional parameters
    sample_array, _, _ = xr.broadcast(
        sample_array, gmm_database["site"], gmm_database["period"]
    )
    correlation = period_correlation_matrix(gmm_database["period"].values)
    dimensions = ["sample", "site", "period"]
    sample_variate = xr.apply_ufunc(
        draw_random_variate,
        sample_array["z"],
        input_core_dims=[dimensions],
        output_core_dims=[dimensions],
        kwargs=dict(rng=rng, correlation=correlation),
    )

    sample_array["epsilon"] = sample_variate
    return sample_array


def normalise_dtypes(dset: xr.Dataset) -> xr.Dataset:
    # normalise dtypes so that indexing is possible with 32-bit floats
    dset["z"] = dset["z"].astype(np.float32)
    dset["period"] = dset["period"].astype(np.float32)
    return dset


@app.command()
def monte_carlo_hazard(
    ruptures: Path,
    ground_motion_database: Path,
    n: int,
    gmm_hazard_path: Path,
    num_realisations: int = 100,
    seed: int = 0,
    column: str = "kl_density",
) -> None:
    ruptures = gpd.read_parquet(ruptures).rename_axis("rupture")
    with xr.open_dataset(ground_motion_database, mode="r") as gmm_database:
        gmm_database = normalise_dtypes(gmm_database)
        thresholds_arr = THRESHOLDS
        thresholds_da = np.log(singleton_array(thresholds_arr, "threshold"))

        # Collect ensemble results
        all_hazard_results = []
        logic_tree = logic_tree_with_weights()
        # Range is from seed to seed + num_realisations to ensure independence
        rng = np.random.default_rng(seed)
        pbar = tqdm.trange(num_realisations, unit="realisation")
        for i in pbar:
            sample_array = draw_sample_inputs(
                n, ruptures[column].to_xarray(), logic_tree, gmm_database, rng=rng
            )
            samples = sample_ground_motions(sample_array, gmm_database)

            rate = ruptures["rate"].to_xarray()
            poe = (samples > thresholds_da).mean("sample")
            hazard = (rate * poe).sum("rupture")
            all_hazard_results.append(hazard.compute())
            mem = psutil.Process().memory_info().rss
            mem_gb = mem / 1e09
            pbar.set_description(f"Memory usage ({mem_gb:.2f} GB)")

    rupture_hazard_ensemble = xr.concat(
        all_hazard_results,
        dim=pd.Index(range(num_realisations), name="realisation"),
    )
    rupture_hazard_ensemble.attrs["seed"] = seed

    rupture_hazard_ensemble.to_zarr(gmm_hazard_path, mode="w")


if __name__ == "__main__":
    app()
