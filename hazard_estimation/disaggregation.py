import functools
from pathlib import Path

import cyclopts
import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import shapely
import xarray as xr
from geocube.api.core import make_geocube

from hazard_estimation import psha

app = cyclopts.App()


def generate_source_model(ruptures: pd.DataFrame) -> psha.SourceModel:
    area = ruptures["area"]
    rake = ruptures["rake"]
    rate = ruptures["rate"]
    magnitude, magnitude_std = psha.get_leonard_magnitude_params(
        area.values / 1e6, rake.values
    )
    return psha.SourceModel(
        rates=rate, mean_magnitudes=magnitude, stddev_magnitudes=magnitude_std
    )


def generate_source_to_site(source_to_site_df: pd.DataFrame) -> psha.SourceToSite:
    return psha.SourceToSite(rrup=source_to_site_df["rrup"])


def hazard_thresholds(composite_hazard: xr.Dataset, rates: np.ndarray) -> xr.DataArray:
    rupture_hazard = composite_hazard["hazard"].sum("rupture")
    total_hazard = rupture_hazard + composite_hazard["ds_hazard"]
    rate_array = xr.DataArray(rates, dims=("rate",), coords=dict(rate=rates))
    threshold = np.abs(total_hazard - rate_array).idxmin("threshold")
    return threshold


def optimal_hazard_sampling_densities(
    rupture_hazard: xr.DataArray, rate: pd.Series
) -> xr.DataArray:
    density = np.sqrt(rate.to_xarray() * rupture_hazard)
    density /= density.sum("rupture")
    return density


def period_independent_population_and_ds_weighted_sampling(
    composite_hazard: xr.Dataset,
    sites: gpd.GeoDataFrame,
    ruptures: gpd.GeoDataFrame,
    rates: np.ndarray,
) -> gpd.GeoDataFrame:
    thresholds = hazard_thresholds(composite_hazard, rates)
    rupture_hazard = composite_hazard.hazard.sel(threshold=thresholds)
    breakpoint()
    sampling_density = optimal_hazard_sampling_densities(
        rupture_hazard, ruptures["rate"]
    )

    ds_hazard = composite_hazard.ds_hazard.sel(threshold=thresholds)
    ds_contribution = ds_hazard / (
        rupture_hazard.sum("rupture") + ds_hazard
    )  # (period, site)
    sites["cell_population"] = sites["cell_population"].fillna(0)
    population_density = sites["cell_population"] / sites["cell_population"].sum()
    weights = (1 - ds_contribution) * population_density.to_xarray()  # (period, site)
    weights /= weights.sum()
    sampling_density = (weights * sampling_density).sum(["period", "site"])
    ruptures = ruptures.copy()
    ruptures["kl_density"] = sampling_density.to_series()
    ruptures["rate_density"] = ruptures["rate"] / ruptures["rate"].sum()

    return ruptures


@app.command()
def run_sampling_from_paths(
    hazard_path: Path,
    sites_path: Path,
    ruptures_path: Path,
    target_rate: float,
    output_path: Path,
) -> None:
    """
    Wrapper to load datasets from disk and compute population/DS weighted sampling.
    """
    # 1. Load data
    sites = gpd.read_parquet(sites_path)
    sites.index.rename("site", inplace=True)
    ruptures = gpd.read_parquet(ruptures_path)
    ruptures.index.rename("rupture", inplace=True)
    with xr.open_dataset(hazard_path, engine="h5netcdf") as composite_hazard:
        # 2. Call the core logic function
        sampled_ruptures = period_independent_population_and_ds_weighted_sampling(
            composite_hazard, sites, ruptures, target_rate
        )

    sampled_ruptures.to_parquet(output_path)


NZ_COASTLINE_URL = "https://www.dropbox.com/scl/fi/zkohh794y0s2189t7b1hi/NZ.gmt?rlkey=02011f4morc4toutt9nzojrw1&st=vpz2ri8x&dl=1"
KNOWN_HASH = "31660def8f51d6d827008e6f20507153cfbbfbca232cd661da7f214aff1c9ce3"


def get_nz_geodataframe() -> gpd.GeoDataFrame:
    file_path = pooch.retrieve(
        NZ_COASTLINE_URL,
        KNOWN_HASH,
    )

    gdf = gpd.read_file(file_path).set_crs(4326, allow_override=True).to_crs(2193)
    gdf["geometry"] = gdf["geometry"].apply(lambda g: shapely.polygonize([g]).geoms[0])
    return gdf


def spatial_density(
    gdf: gpd.GeoDataFrame, geometry_gdf: gpd.GeoDataFrame | None = None
):
    if geometry_gdf is None:
        geometry_gdf = get_nz_geodataframe()

    geometry = shapely.MultiPolygon(geometry_gdf["geometry"])
    voronoi_diagram = gpd.GeoDataFrame(
        dict(geometry=gdf.voronoi_polygons(extend_to=geometry))
    ).clip(geometry)

    gdf = gdf.sjoin(voronoi_diagram, how="left")
    gdf["cell"] = voronoi_diagram["geometry"].loc[gdf["index_right"]].values
    gdf = gdf.drop(columns=["index_right"])
    area = gdf["cell"].area
    total = voronoi_diagram.area.sum()
    density = area / total
    gdf["density"] = density
    return gdf


def population_density(
    gdf: gpd.GeoDataFrame,
    population_blocks: gpd.GeoDataFrame,
    block_resolution: float = 250**2,
    population_column: str = "PopEst2023",
) -> gpd.GeoDataFrame:
    block_resolved = (
        gdf.reset_index()
        .set_geometry("cell")
        .overlay(population_blocks, how="intersection")
    )
    population_in_cells = block_resolved.groupby("site")[population_column].sum()
    gdf["population_density"] = population_in_cells / population_in_cells.sum()
    # Some cells have no population blocks in them, we assume no population here.
    gdf["population_density"] = gdf["population_density"].fillna(0.0)
    return gdf


def _rasterize_batch(
    batch_df: np.ndarray, column: str, crs: str, master_grid_coords: xr.Dataset
) -> np.ndarray:
    """
    Worker function: Rasterizes a batch of ruptures and sums them locally.
    Returns a single numpy array to minimize IPC memory overhead.
    """
    # Initialize the accumulator for this batch
    shape = (len(master_grid_coords.y), len(master_grid_coords.x))
    batch_sum = np.zeros(shape, dtype=np.float32)

    for row in batch_df:
        # Wrap row in GDF for geocube
        single_gdf = gpd.GeoDataFrame([row], crs=crs)

        raster = make_geocube(
            vector_data=single_gdf,
            measurements=[column],
            like=master_grid_coords,
            fill=0.0,
        )
        # Add to local batch sum and immediately allow temporary raster to be GC'd
        batch_sum += raster[column].values.astype(np.float32)

    return batch_sum


from geocube.rasterize import rasterize_image
from rasterio.enums import MergeAlg


def patch_density_raster(
    ruptures_df: gpd.GeoDataFrame,
    column: str,
    resolution: float = 500.0,
) -> xr.DataArray:
    """
    Fast, vectorized rasterization that sums overlapping polygons.
    Avoids iteration and multiprocessing entirely.
    """
    # Use functools.partial to bake the merge_alg into the default rasterizer
    custom_rasterize = functools.partial(rasterize_image, merge_alg=MergeAlg.add)

    master_grid = make_geocube(
        vector_data=ruptures_df,
        measurements=[column],
        resolution=(-resolution, resolution),
        fill=0.0,
        rasterize_function=custom_rasterize,
    )

    return master_grid[column]


def kl_centroid(
    ruptures: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    disagg: xr.Dataset,
    column: str = "density",
):
    # Trim stations with high distributed seismicity contribution
    disagg = disagg.sel(site=sites.index)
    disagg = disagg.sel(site=(disagg.ds_hazard < 0.98))
    sites = sites.loc[disagg.site.values]
    ps = [
        optimal_proposal_distribution(ruptures, disagg.disagg.sel(site=site))
        for site in disagg.site.values
    ]

    kl_centroid = pd.Series(
        np.average(ps, axis=0, weights=sites[column]), index=ruptures.index
    )
    return kl_centroid


def generate_multinomial_map(gdf, planes_df, weight_col, n_samples, cmap="hot"):
    counts = np.random.multinomial(n_samples, gdf[weight_col])

    sampled_gdf = gdf.copy()
    sampled_gdf["sample_count"] = counts
    sampled_gdf = patch_density(sampled_gdf, planes_df, "sample_count")
    m = sampled_gdf.explore(
        column="sample_count",
        cmap=cmap,
        legend=True,
        tooltip=["sample_count"],
        # vmin/vmax should be adjusted for counts (1 to ~10 or higher)
        vmin=1,
        vmax=10,
    )

    return m


if __name__ == "__main__":
    app()
