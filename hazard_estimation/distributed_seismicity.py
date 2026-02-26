from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from hazard_estimation.site import Site


def get_hazard_at(
    seismicity_model_grouped: pd.api.typing.DataFrameGroupBy,
    site: Site,
    period: float,
    threshold: float,
) -> float:
    seismicity_model = gpd.GeoDataFrame(seismicity_model_grouped.get_group(period))
    closest_site = seismicity_model.sindex.nearest(shapely.Point(site.lon, site.lat))[
        -1
    ]
    local_seismicity_model = seismicity_model.loc[closest_site].sort_values("threshold")
    return np.interp(
        threshold, local_seismicity_model["threshold"], local_seismicity_model["rate"]
    )


def load_seismicity_model(seismicity_path: Path) -> gpd.GeoDataFrame:
    # first row is OQ metadata
    df = pd.read_csv(seismicity_path, skiprows=1)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"])
    ).set_crs(4326)
    values = [column for column in gdf.columns if column.startswith("poe-")]
    gdf = gdf.melt(
        id_vars="geometry", value_vars=values, var_name="threshold", value_name="poe"
    )
    gdf["rate"] = -np.log(np.maximum(1 - gdf["poe"], 1e-6)) / 50
    gdf["threshold"] = gdf["threshold"].str.removeprefix("poe-").astype(float)
    return gpd.GeoDataFrame(gdf[["geometry", "threshold", "rate"]], crs=4326)
