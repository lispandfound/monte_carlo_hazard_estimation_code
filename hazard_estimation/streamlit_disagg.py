import argparse

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from matplotlib import cm

from hazard_estimation import disaggregation


def create_density_layer(
    map, da, name, vmin=-8, vmax=-1, cmap_name="magma", log: bool = False
):
    """
    Returns a folium.FeatureGroup containing a Raster layer and its Legend.
    """
    # 1. Coordinate Prep
    da_4326 = da.rio.reproject("EPSG:4326")

    # 2. Log Data Transformation
    data = da_4326.values
    if log:
        data[data <= 0] = np.nan
        data = np.log10(data)
    data_clipped = np.clip(data, vmin, vmax)

    # 3. Create the Color Scale
    cmap = cm.get_cmap(cmap_name)
    colors_hex = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, 10)]

    legend = bcm.LinearColormap(
        colors=colors_hex, vmin=vmin, vmax=vmax, caption=f"{name}"
    )

    # 4. Prepare the Image
    norm_data = (data_clipped - vmin) / (vmax - vmin)
    rgba_image = cmap(norm_data)
    rgba_image[np.isnan(data)] = [0, 0, 0, 0]

    # 5. Build the Layer Group

    bounds = [
        [float(da_4326.y.min()), float(da_4326.x.min())],
        [float(da_4326.y.max()), float(da_4326.x.max())],
    ]

    overlay = folium.raster_layers.ImageOverlay(
        image=rgba_image, bounds=bounds, name=name, opacity=0.7, interactive=True
    )
    overlay.add_to(map)

    legend.add_to(map)


def rate_map(
    ruptures_a: gpd.GeoDataFrame,
    ruptures_b: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
) -> folium.Map:

    kl_a_raster = disaggregation.patch_density_raster(
        ruptures_a, "kl_density", resolution=1000
    )
    kl_b_raster = disaggregation.patch_density_raster(
        ruptures_b, "kl_density", resolution=1000
    )

    diff_raster = kl_b_raster / kl_a_raster

    # 2. Site Preparation
    # Ensure sites is in 4326 for Folium
    sites_mapped = sites.to_crs(epsg=4326)

    # Map ds_hazard from xarray back to the GeoDataFrame
    # Assumes 'station' or index in sites matches the dimension name in disagg

    # Define a color map for the kl density weighting
    # We use a viridis or similar scale to differentiate from the 'hot' rasters
    site_cm = bcm.linear.YlGnBu_09.scale(
        sites_mapped["density"].min(), sites_mapped["density"].max()
    )

    m = folium.Map(location=[-41.0, 174.0], zoom_start=6)

    # 3. Add Rasters
    create_density_layer(
        m,
        kl_a_raster,
        "KL-weighted sampling (a, log10)",
        vmin=-8,
        vmax=0,
        cmap_name="hot",
        log=True,
    )
    create_density_layer(
        m,
        kl_b_raster,
        "KL-weighted sampling (b, log10)",
        vmin=-8,
        vmax=0,
        cmap_name="hot",
        log=True,
    )
    create_density_layer(
        m,
        diff_raster,
        "Sampling difference (log10)",
        vmin=-0.3,
        vmax=0.3,
        cmap_name="RdBu",
        log=True,
    )

    # 4. Add Sites with Custom Styling and Tooltip
    folium.GeoJson(
        sites_mapped.set_geometry("geometry").reset_index()[["station", "geometry"]],
        name="GeoNet Sites (KL weighted)",
        marker=folium.CircleMarker(
            radius=4,
            weight=1,
            fill=True,
            fill_color="white",
            color="black",
            fill_opacity=0.8,
        ),
        tooltip=folium.GeoJsonTooltip(
            fields=["station"],
            aliases=["Name"],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
            """,
            max_width=800,
        ),
    ).add_to(m)

    # Add the Site Legend specifically to the map
    site_cm.caption = "Site KL Density Weighting"
    m.add_child(site_cm)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hazard Estimation Disaggregation Maps"
    )

    # File Inputs
    parser.add_argument(
        "ruptures_a", type=str, help="Path to ruptures GeoPackage/Shapefile"
    )
    parser.add_argument(
        "ruptures_b", type=str, help="Path to ruptures GeoPackage/Shapefile"
    )

    parser.add_argument("sites", type=str, help="Path to sites GeoPackage/Shapefile")

    # Output
    parser.add_argument(
        "--output", type=str, default="hazard_map.html", help="Output HTML filename"
    )

    # Sampling/Subsetting
    parser.add_argument(
        "--sample-sites",
        type=int,
        default=None,
        help="Number of sites to sample (for performance)",
    )
    parser.add_argument(
        "--shape", type=str, default=None, help="Outline to clip polygon"
    )

    args = parser.parse_args()

    ruptures_a_gdf = gpd.read_parquet(args.ruptures_a)
    ruptures_b_gdf = gpd.read_parquet(args.ruptures_b)
    sites_gdf = gpd.read_parquet(args.sites)

    m = rate_map(
        ruptures_a_gdf,
        ruptures_b_gdf,
        sites_gdf,
    )

    m.save(args.output)


if __name__ == "__main__":
    main()
