import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from hazard_estimation.site import Site
import shapely


def get_hazard_at(seismicity_model: gpd.GeoDataFrame, site: Site, threshold: float) -> float:
    closest_site = seismicity_model.sindex.nearest(shapely.Point(site.lon, site.lat))[-1]
    local_seismicity_model = seismicity_model.loc[closest_site].sort_values('threshold')
    return np.interp(threshold, local_seismicity_model['threshold'], local_seismicity_model['rate']) 


def load_seismicity_model(seismicity_path: Path) -> gpd.GeoDataFrame:
    # first row is OQ metadata
    df = pd.read_csv(seismicity_path, skiprows=1)    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat'])).set_crs(4326)
    values = [column for column in gdf.columns if column.startswith('poe-')]
    gdf = gdf.melt(id_vars='geometry', value_vars=values, var_name='threshold', value_name='poe')
    gdf['rate'] = -np.log(np.maximum(1 - gdf['poe'], 1e-6)) / 50
    gdf['threshold'] = gdf['threshold'].str.removeprefix('poe-').astype(float)
    return gpd.GeoDataFrame(gdf[['geometry', 'threshold', 'rate']], crs=4326)

    




