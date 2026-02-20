
import branca.colormap as cm
import cyclopts
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import scipy as sp
import shapely
import tqdm
import xarray as xr
from geocube.api.core import make_geocube
from tqdm import tqdm # Optional: for a progress bar

from hazard_estimation import psha, distributed_seismicity
from hazard_estimation.site import Site

from tqdm.contrib.concurrent import process_map

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


def disaggregate_ruptures(
    ruptures: pd.DataFrame,
    source_to_site_df: pd.DataFrame,
    site: Site,
    ds_model: gpd.GeoDataFrame,
    hazard_rate: float,
) -> pd.Series:
    joint_df = ruptures.join(source_to_site_df["rrup"])
    source_model = generate_source_model(joint_df)
    source_to_site = generate_source_to_site(joint_df)

    def hazard_objective(threshold: float, target: float) -> float:
        ds_hazard = distributed_seismicity.get_hazard_at(ds_model, site, threshold)
        hazard = psha.analytical_hazard(source_model, site, source_to_site, threshold)
        return hazard.sum() + ds_hazard - target

    threshold = sp.optimize.bisect(
        hazard_objective, 0, 2.0, rtol=1e-3, args=(hazard_rate,)
    )
    hazard = psha.analytical_hazard(source_model, site, source_to_site, threshold)
    ds_hazard = distributed_seismicity.get_hazard_at(ds_model, site, threshold)

    return threshold, ds_hazard / hazard_rate, pd.Series(hazard, index=joint_df.index)


def _disaggregate_single_site_worker(site_info, ruptures, source_to_site_group, ds_model, hazard_rate):
    """Worker function to process a single site."""
    name, properties = site_info
    
    cur_site = psha.Site(
        name,
        lon=properties["geometry"].x,
        lat=properties["geometry"].y,
        vs30=properties["vs30"],
    )
    
    # We pass only the specific group for this site to avoid huge data transfers
    threshold, ds_hazard, site_disagg = disaggregate_ruptures(
        ruptures, source_to_site_group, cur_site, ds_model, hazard_rate
    )
    
    # Return the results as a tuple to be reassembled
    return threshold, ds_hazard, site_disagg.values

def disaggregate_all_sites(
    ruptures, source_to_site_df, sites, ds_model, hazard_rate, n_procs=None
) -> xr.Dataset:
    
    grouped = source_to_site_df.groupby("site")
    
    args = [
        (
            (name, props), 
            ruptures, 
            grouped.get_group(name), 
            ds_model, 
            hazard_rate
        ) 
        for name, props in sites.iterrows()
    ]

    results = process_map(
        _disaggregate_single_site_worker,
        *zip(*args),
        max_workers=n_procs,
        chunksize=1,
        unit="site",
        desc="Disaggregating Sites"
    )

    thresholds, ds_hazards, disagg_list = zip(*results)
    
    disagg_matrix = np.vstack(disagg_list)

    array = xr.Dataset(
        data_vars=dict(
            disagg=(("site", "rupture"), disagg_matrix),
            threshold=(("site",), list(thresholds)),
            ds_hazard=(("site",), list(ds_hazards)),
        ),
        coords=dict(
            site=sites.index.values,
            rupture=ruptures.index.values,
        ),
    )
    return array


def optimal_proposal_distribution(
    ruptures: pd.DataFrame, disagg: xr.DataArray
) -> pd.Series:
    hazard = disagg.to_dataframe()["disagg"]
    rate = ruptures["rate"]
    p = np.sqrt(rate * hazard)
    p /= p.sum()
    return p


def visualise_distribution(ruptures: gpd.GeoDataFrame, distribution: xr.DataArray):
    """
    Visualises rupture geometries colored by hazard values in log10 space.
    """
    # 1. Ensure the GeoDataFrame is in WGS84 for Folium
    ruptures = ruptures.to_crs(epsg=4326)

    # 2. Align the disagg data to the ruptures GeoDataFrame
    # We assume 'disagg' shares the same index/order as 'ruptures'
    df = ruptures.copy()
    df["p"] = distribution
    # 3. Apply log-space transformation
    # We use a small epsilon to avoid log(0) errors
    epsilon = 1e-12
    df["log_p"] = np.log10(df["p"] + epsilon)
    df["log_rate"] = np.log10(df["rate"])

    # Using 'YlOrRd' (Yellow-Orange-Red) for p visualization
    colormap = cm.LinearColormap(
        colors=["blue", "cyan", "yellow", "orange", "red"],
        vmin=-12,
        vmax=0,
        caption="P Level (log10 scale)",
    )

    # 5. Initialize the Map
    # Center on the centroid of the geometries
    center = [df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

    # 6. Define Style Function
    def style_fn(feature):
        log_val = feature["properties"]["log_p"]
        return {
            "fillColor": colormap(log_val),
            "color": colormap(log_val),
            "weight": 2,
            "fillOpacity": 0.6,
        }

    # 7. Add GeoJSON to Map
    folium.GeoJson(
        df,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["log_rate", "log_p"],
            aliases=["Log10 Rate:", "Log10 P:"],
            localize=True,
        ),
    ).add_to(m)

    m.add_child(colormap)
    return m


def visualise_disagg(ruptures: gpd.GeoDataFrame, disagg: xr.DataArray):
    """
    Visualises rupture geometries colored by hazard values in log10 space.
    """
    # 1. Ensure the GeoDataFrame is in WGS84 for Folium
    ruptures = ruptures.to_crs(epsg=4326)

    # 2. Align the disagg data to the ruptures GeoDataFrame
    # We assume 'disagg' shares the same index/order as 'ruptures'
    df = ruptures.copy()
    df["hazard"] = disagg.to_dataframe()["disagg"]
    # 3. Apply log-space transformation
    # We use a small epsilon to avoid log(0) errors
    epsilon = 1e-12
    df["log_hazard"] = np.log10(df["hazard"] + epsilon)
    df["log_rate"] = np.log10(df["rate"])

    # Using 'YlOrRd' (Yellow-Orange-Red) for hazard visualization
    colormap = cm.LinearColormap(
        colors=["blue", "cyan", "yellow", "orange", "red"],
        vmin=-12,
        vmax=-3,
        caption="Hazard Level (log10 scale)",
    )

    # 5. Initialize the Map
    # Center on the centroid of the geometries
    center = [df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

    # 6. Define Style Function
    def style_fn(feature):
        log_val = feature["properties"]["log_hazard"]
        return {
            "fillColor": colormap(log_val),
            "color": colormap(log_val),
            "weight": 2,
            "fillOpacity": 0.6,
        }

    # 7. Add GeoJSON to Map
    folium.GeoJson(
        df,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["log_rate", "log_hazard"],
            aliases=["Log10 Rate:", "Log10 Hazard:"],
            localize=True,
        ),
    ).add_to(m)

    m.add_child(colormap)
    return m


NZ_COASTLINE_URL = "https://www.dropbox.com/scl/fi/zkohh794y0s2189t7b1hi/NZ.gmt?rlkey=02011f4morc4toutt9nzojrw1&st=vpz2ri8x&dl=1"
KNOWN_HASH = "31660def8f51d6d827008e6f20507153cfbbfbca232cd661da7f214aff1c9ce3"


def get_nz_geodataframe() -> gpd.GeoDataFrame:
    file_path = pooch.retrieve(
        NZ_COASTLINE_URL,
        KNOWN_HASH,
    )

    gdf = gpd.read_file(file_path).set_crs(4326, allow_override=True).to_crs(2193)
    gdf['geometry'] = gdf['geometry'].apply(lambda g: shapely.polygonize([g]).geoms[0])
    return gdf





def spatial_density(gdf: gpd.GeoDataFrame, geometry_gdf: gpd.GeoDataFrame | None = None):
    if geometry_gdf is None:
        geometry_gdf = get_nz_geodataframe()

    geometry = shapely.MultiPolygon(geometry_gdf['geometry'])
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

def patch_density_raster(ruptures_df: gpd.GeoDataFrame, column: str, resolution: float = 500.0) -> xr.DataArray:
    """
    Iterates over ruptures, rasterizes each, and sums them into a master DataArray.
    """
    # 1. Initialize the master grid using the total bounds of all ruptures
    # We create a 'like' template so all subsequent rasters have the same shape/coords
    template_gdf = gpd.GeoDataFrame({'dummy': [0]}, geometry=[ruptures_df.unary_union.envelope], crs=ruptures_df.crs)
    
    master_grid = make_geocube(
        vector_data=template_gdf,
        measurements=['dummy'],
        resolution=(-resolution, resolution),
        fill=0.0
    )
    
    # Extract the DataArray and ensure it's float32 to save memory
    full_density = xr.zeros_like(master_grid['dummy'], dtype=np.float32)

    # 2. Iterate and Accumulate
    # We use MergeAlg.add to ensure overlapping parts of a MultiPolygon 
    # within a SINGLE rupture are summed, then we add that to the master.
    for _, row in tqdm(ruptures_df.iterrows(), total=len(ruptures_df), desc="Rasterizing"):
        # Convert single row back to a GDF for geocube
        single_rupture_gdf = gpd.GeoDataFrame([row], crs=ruptures_df.crs)
        
        # Rasterize just this one rupture
        one_rupture_raster = make_geocube(
            vector_data=single_rupture_gdf,
            measurements=[column],
            like=master_grid, # Force alignment to master grid
            fill=0.0,
        )
        
        # Add this rupture's density to the master accumulator
        full_density += one_rupture_raster[column]

    return full_density


def kl_centroid(ruptures: gpd.GeoDataFrame, sites: gpd.GeoDataFrame, disagg: xr.Dataset):
    # Trim stations with high distributed seismicity contribution
    disagg = disagg.sel(site=sites.index)
    disagg = disagg.sel(site=(disagg.ds_hazard < 0.98))
    sites = sites.loc[disagg.site.values]
    breakpoint()
    ps = [
        optimal_proposal_distribution(ruptures, disagg.disagg.sel(site=site))
        for site in disagg.site.values
    ]

    kl_centroid = pd.Series(
        np.average(ps, axis=0, weights=sites["density"]), index=ruptures.index
    )
    return kl_centroid





def generate_multinomial_map(gdf, planes_df, weight_col, n_samples, cmap="hot"):
    counts = np.random.multinomial(n_samples, gdf[weight_col])

    sampled_gdf = gdf.copy()
    sampled_gdf["sample_count"] = counts
    sampled_gdf = patch_density(sampled_gdf, planes_df, 'sample_count')
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
