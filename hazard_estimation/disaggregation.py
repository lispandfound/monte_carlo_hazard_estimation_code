import argparse

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy as sp
import tqdm
import xarray as xr

from hazard_estimation import psha
from hazard_estimation.site import Site


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
    hazard_rate: float,
) -> pd.Series:
    joint_df = ruptures.join(source_to_site_df["rrup"])
    source_model = generate_source_model(joint_df)
    source_to_site = generate_source_to_site(joint_df)

    def hazard_objective(threshold: float, target: float) -> float:
        hazard = psha.analytical_hazard(source_model, site, source_to_site, threshold)
        return hazard.sum() - target

    threshold = sp.optimize.bisect(
        hazard_objective, 0, 2.0, rtol=1e-3, args=(hazard_rate,)
    )
    hazard = psha.analytical_hazard(source_model, site, source_to_site, threshold)

    return threshold, pd.Series(hazard, index=joint_df.index).sort_values()


def disaggregate_all_sites(
    ruptures: pd.DataFrame,
    source_to_site_df: pd.DataFrame,
    sites: gpd.GeoDataFrame,
    hazard_rate: float,
) -> xr.Dataset:
    thresholds = []
    index = ruptures.index
    disagg = np.zeros((len(sites), len(ruptures)))
    for i, (name, properties) in tqdm.tqdm(
        enumerate(sites.iterrows()), total=len(sites), unit="site"
    ):
        cur_site = psha.Site(
            name,
            lon=properties["geometry"].x,
            lat=properties["geometry"].y,
            vs30=properties["vs30"],
        )
        source_to_site = source_to_site_df.groupby("site").get_group(name)
        threshold, site_disagg = disaggregate_ruptures(
            ruptures, source_to_site, cur_site, hazard_rate
        )
        disagg[i] = site_disagg.loc[index]
        thresholds.append(threshold)

    array = xr.Dataset(
        dict(disagg=(("site", "rupture"), disagg), threshold=(("site", thresholds))),
        coords=dict(
            site=sites.index.values,
            rupture=index.values,
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
