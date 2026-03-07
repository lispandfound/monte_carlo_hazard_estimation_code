import bisect
import functools
import itertools
from pathlib import Path

import cyclopts
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import seaborn as sns
import shapely
import tqdm
import xarray as xr
from geocube.api.core import make_geocube
from matplotlib import pyplot as plt

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


def hazard_thresholds(composite_hazard: xr.Dataset, target_rate: float) -> xr.DataArray:
    rupture_hazard = composite_hazard["hazard"].sum("rupture")
    total_hazard = rupture_hazard + composite_hazard["ds_hazard"]
    threshold = np.abs(total_hazard - target_rate).idxmin("threshold")
    return threshold


def optimal_hazard_sampling_densities(
    rupture_hazard: xr.DataArray, rate: pd.Series, random: bool
) -> xr.DataArray:
    if random:
        density = np.sqrt(rate.to_xarray() * rupture_hazard)
    else:
        density = np.sqrt(rate.to_xarray() * rupture_hazard - rupture_hazard**2)
    density /= density.sum("rupture")
    return density


def period_independent_population_and_ds_weighted_sampling(
    composite_hazard: xr.Dataset,
    sites: gpd.GeoDataFrame,
    ruptures: gpd.GeoDataFrame,
    rates: float,
    periods: np.ndarray,
    random: bool = False,
) -> xr.DataArray:
    thresholds = hazard_thresholds(composite_hazard, rates)
    rupture_hazard = composite_hazard.hazard.sel(threshold=thresholds).sel(
        period=periods, method="nearest"
    )
    sampling_density = optimal_hazard_sampling_densities(
        rupture_hazard, ruptures["rate"], random=random
    )

    ds_hazard = composite_hazard.ds_hazard.sel(threshold=thresholds).sel(
        period=periods, method="nearest"
    )
    ds_contribution = ds_hazard / (
        rupture_hazard.sum("rupture") + ds_hazard
    )  # (period, site)
    sites["cell_population"] = sites["cell_population"].fillna(0)
    population_density = sites["cell_population"] / sites["cell_population"].sum()
    weights = (1 - ds_contribution) * population_density.to_xarray()  # (period, site)
    weights /= weights.sum()
    sampling_density = (weights * sampling_density).sum(["period", "site"])
    return sampling_density


@app.command()
def run_sampling_from_paths(
    hazard_path: Path,
    sites_path: Path,
    ruptures_path: Path,
    target_probs: list[float],
    output: Path | None = None,
    periods: list[float] | None = None,
    random: bool = False,
) -> None:
    """
    Wrapper to load datasets from disk and compute population/DS weighted sampling.
    """
    # 1. Load data
    sites = gpd.read_parquet(sites_path)
    sites.index.rename("site", inplace=True)
    ruptures = gpd.read_parquet(ruptures_path)
    ruptures.index.rename("rupture", inplace=True)
    target_rates = -np.log(1 - np.array(target_probs)) / 50
    with xr.open_dataset(hazard_path, engine="h5netcdf") as composite_hazard:
        periods_arr = np.array(periods) if periods else composite_hazard.period.values
        density = np.zeros((len(target_rates), len(periods_arr) + 1, len(ruptures)))
        sampling_periods = [periods_arr] + [
            periods_arr[[i]] for i in range(periods_arr.size)
        ]
        density_da = xr.DataArray(
            density,
            dims=("rate", "period", "rupture"),
            coords=dict(
                period=[p.mean() for p in sampling_periods],
                rate=target_rates,
                rupture=ruptures.index,
            ),
        )
        pairs = list(
            itertools.product(range(len(target_rates)), range(len(sampling_periods)))
        )
        for i, j in tqdm.tqdm(pairs, unit="pair"):
            rate = target_rates[i]
            array = period_independent_population_and_ds_weighted_sampling(
                composite_hazard,
                sites,
                ruptures,
                rate,
                sampling_periods[j],
                random=random,
            )
            density_da[i][j] = array

    density_da.to_netcdf(output, engine="h5netcdf")


def random_sampling_variance(
    sampling_densities: xr.DataArray, rupture_rates: xr.DataArray, hazard: xr.DataArray
):
    total_hazard = hazard.sum().item()
    np.sqrt(rupture_rates * hazard).sum()


@app.command()
def evaluate_monte_carlo_sample(
    monte_carlo_hazard_path: Path,
    ruptures_path: Path,
    site_path: Path,
    hazard_path: Path,
    output_path: Path,
    sites: list[str],
):
    ruptures = gpd.read_parquet(ruptures_path).rename_axis("rupture")
    rates = ruptures["rate"].to_xarray()
    with (
        xr.open_dataset(hazard_path, engine="h5netcdf") as composite_hazard,
        xr.open_dataset(
            monte_carlo_hazard_path, engine="h5netcdf"
        ) as monte_carlo_hazard,
    ):
        error = experimental_error(monte_carlo_hazard, rates, composite_hazard)

    output_path.mkdir(exist_ok=True, parents=True)
    plot_site_performance_to(error.sel(site=sites), output_path, empirical=True)
    sites_gdf = gpd.read_parquet(site_path)
    m = spatial_error_map(error, sites_gdf)
    m.save(output_path / "error_map.html")


@app.command()
def evaluate_sampling_strategy(
    sampling_densities_path: Path,
    hazard_path: Path,
    output_path: Path,
    random: bool = False,
):
    with (
        xr.open_dataset(hazard_path, engine="h5netcdf") as composite_hazard,
        xr.open_dataset(
            sampling_densities_path, engine="h5netcdf"
        ) as sampling_densities,
    ):
        total_hazard = composite_hazard.sum("rupture")
        for site in sampling_densities.site:
            m = range(len(sampling_densities.period))
            n = range(len(sampling_densities.rate))
            bias = np.zeros(sampling_densities.shape, dtype=float)
            variance = np.zeros_like(bias)
            ess = np.zeros_like(bias)
            error_dset = xr.Dataset(
                dict(
                    bias=(("rate", "period", "threshold"), bias),
                    variance=(("rate", "period", "threshold"), variance),
                    ess=(("rate", "period", "threshold"), ess),
                ),
                coords=dict(
                    rate=sampling_densities.rate, period=sampling_densities.period
                ),
            )
            total_hazard = composite_hazard.sel(site=site).sum("rupture")
            for i, j in itertools.product(*error_dset.bias.shape):
                if random:
                    variance = random_sampling_variance


def plot_kl_divergence_period_heatmap(da: xr.DataArray):
    Q = da.isel(period=0)

    P = da.isel(period=slice(1, None))

    epsilon = 1e-12
    kl_div = (P * np.log((P + epsilon) / (Q + epsilon))).sum(dim="rupture")

    kl_div = kl_div.sortby("period")

    df_plot = kl_div.to_pandas().T

    fig, ax = plt.subplots(figsize=(12, 7))

    xticklabels = [f"{r:.2e}" for r in df_plot.columns]
    yticklabels = [f"{p:.2f}" for p in df_plot.index]

    sns.heatmap(
        df_plot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "KL Divergence (bits)"},
        yticklabels=yticklabels,
        xticklabels=xticklabels,
        ax=ax,
    )

    ax.set_title(
        "Sensitivity: KL Divergence of Period-Specific Optimal vs. Mean Case",
        fontsize=14,
    )
    ax.set_xlabel(r"Target Hazard Rate ($\lambda$)", fontsize=12)
    ax.set_ylabel("Spectral Period (s)", fontsize=12)

    fig.tight_layout()
    return fig


def plot_pairwise_rate_kl_divergence(da: xr.DataArray):
    # 1. Isolate the mean period case (isel=0)
    # Shape is (rate, rupture)
    mean_period_da = da.isel(period=0)

    # 2. Create two versions with different dimension names for broadcasting
    # This lets us compute every pair (rate_p, rate_q)
    P = mean_period_da.rename({"rate": "rate_p"})
    Q = mean_period_da.rename({"rate": "rate_q"})

    # 3. Vectorized Pairwise KL Divergence
    # Resulting shape: (rate_p, rate_q)
    epsilon = 1e-12
    # KL(P || Q) = sum( P * log(P/Q) )
    pairwise_kl = (P * np.log((P + epsilon) / (Q + epsilon))).sum(dim="rupture")

    # 4. Sort both axes so the heatmap is logically ordered by rate
    pairwise_kl = pairwise_kl.sortby("rate_p").sortby("rate_q")

    # 5. Convert to pandas for Seaborn
    df_plot = pairwise_kl.to_pandas()

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Format labels as scientific notation for the rates
    labels = [f"{r:.2e}" for r in df_plot.index]

    sns.heatmap(
        df_plot,
        annot=True,
        fmt=".4f",
        cmap="rocket",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "KL Divergence $D_{KL}(Rate_Y || Rate_X)$"},
        ax=ax,
    )

    ax.set_title(
        "Pairwise Rate Sensitivity: $D_{KL}$ between Target Hazard Rates\n(at Mean Period Sampling Density)",
        fontsize=14,
    )
    ax.set_xlabel(r"Reference Rate $Q$ ($\lambda$)", fontsize=12)
    ax.set_ylabel(r"Target Rate $P$ ($\lambda$)", fontsize=12)

    fig.tight_layout()
    return fig


#


def mean_squared_error(
    sampling_densities: xr.DataArray,
    rates: xr.DataArray,
    hazard: xr.Dataset,
    n: int,
) -> xr.Dataset:
    lambda_i = hazard.hazard
    lambda_rup_i = rates
    total_hazard = lambda_i.sum("rupture") + hazard.ds_hazard
    n_i = np.round(n * sampling_densities)
    # Expected value of estimator is total hazard - hazard from ruptures we don't sample
    # by linearity of expectations.
    bias = lambda_i.where(n_i == 0, 0).sum("rupture")
    p_sigma_sq = (lambda_rup_i * lambda_i) - np.square(lambda_i)
    variance = (p_sigma_sq / n_i.where(n_i > 0)).sum("rupture")
    mse = np.square(bias) + variance
    r = len(lambda_i.rupture)
    ess = r * p_sigma_sq.sum("rupture") / mse
    return xr.Dataset(
        dict(hazard=total_hazard, bias=bias, variance=variance, mse=mse, ess=ess),
        attrs=dict(n=n),
    )


def multinomial_error(
    sampling_densities: xr.DataArray,
    rates: xr.DataArray,
    hazard: xr.Dataset,
    n: int,
) -> xr.Dataset:
    # 1. Setup
    lambda_i = hazard.hazard
    lambda_rup_i = rates
    total_lambda_rup = lambda_i.sum("rupture")
    # Dataset total hazard (for reporting)
    total_hazard = total_lambda_rup + hazard.ds_hazard

    term1 = ((lambda_rup_i * lambda_i) / sampling_densities).sum("rupture")
    variance = (1 / n) * (term1 - np.square(total_lambda_rup))

    bias = xr.full_like(variance, 0.0)  # Bias is analytically zero here
    mse = variance  # Since Bias = 0
    p_sigma_sq = (lambda_rup_i * lambda_i) - np.square(lambda_i)
    # ESS based on your factor of R logic
    # ESS = N * (Var_equal / Var_actual)
    r = len(lambda_i.rupture)
    ess = r * p_sigma_sq.sum("rupture") / mse

    return xr.Dataset(
        dict(hazard=total_hazard, bias=bias, variance=variance, mse=mse, ess=ess),
        attrs=dict(n=n),
    )


def experimental_error(
    experimental_hazard: xr.Dataset, rates: xr.DataArray, hazard: xr.Dataset
) -> xr.Dataset:
    total_experimental_hazard = (
        experimental_hazard.hazard + experimental_hazard.ds_hazard
    )
    lambda_i = hazard.hazard
    lambda_rup_i = rates
    total_lambda_rup = lambda_i.sum("rupture")

    total_hazard = total_lambda_rup + hazard.ds_hazard
    se = np.square(total_experimental_hazard - total_hazard)
    mse = se.mean("realisation")
    n_realisations = len(experimental_hazard.realisation)
    mse_se = se.std("realisation") / np.sqrt(n_realisations)
    ci_95 = 1.96 * mse_se
    mse_lower = (mse - ci_95).clip(min=0)
    mse_upper = mse + ci_95

    p_sigma_sq = (lambda_rup_i * lambda_i) - np.square(lambda_i)
    r = len(lambda_i.rupture)
    constant_numerator = r * p_sigma_sq.sum("rupture")
    ess = constant_numerator / mse
    ess_lower = constant_numerator / mse_upper
    # Use xr.where to avoid dividing by 0 if mse_lower is exactly 0
    ess_upper = xr.where(mse_lower > 0, constant_numerator / mse_lower, np.nan)
    return xr.Dataset(
        dict(
            hazard=total_hazard,
            mse=mse,
            mse_lower=mse_lower,
            mse_upper=mse_upper,
            ess=ess,
            ess_lower=ess_lower,
            ess_upper=ess_upper,
        ),
    )


def plot_empirical_performance_metrics(ds_site):
    # 1. Prepare Data
    df = ds_site.to_dataframe().reset_index()

    # Calculate Relative RMSE and bounds (%)
    df["rel_rmse_pct"] = (np.sqrt(df["mse"]) / df["hazard"]) * 100
    df["rel_rmse_pct_lower"] = (np.sqrt(df["mse_lower"]) / df["hazard"]) * 100
    df["rel_rmse_pct_upper"] = (np.sqrt(df["mse_upper"]) / df["hazard"]) * 100

    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    periods = df["period"].unique()
    palette = sns.color_palette("rocket", n_colors=len(periods))

    # 3. Iterate over periods to plot the mean line and shaded confidence intervals
    for i, p in enumerate(periods):
        p_data = df[df["period"] == p].sort_values("hazard", ascending=False)
        color = palette[i]

        # --- Plot 1: Relative RMSE ---
        ax1.plot(
            p_data["hazard"],
            p_data["rel_rmse_pct"],
            color=color,
            alpha=0.9,
            label=f"{p}s",
        )
        ax1.fill_between(
            p_data["hazard"],
            p_data["rel_rmse_pct_lower"],
            p_data["rel_rmse_pct_upper"],
            color=color,
            alpha=0.15,
            edgecolor="none",
        )

        # --- Plot 2: ESS ---
        ax1_line = ax2.plot(
            p_data["hazard"], p_data["ess"], color=color, alpha=0.9, label=f"{p}s"
        )
        # Drop NaNs to prevent matplotlib fill_between artifacts if ess_upper hit infinity
        valid_ess = p_data.dropna(subset=["ess_lower", "ess_upper"])
        ax2.fill_between(
            valid_ess["hazard"],
            valid_ess["ess_lower"],
            valid_ess["ess_upper"],
            color=color,
            alpha=0.15,
            edgecolor="none",
        )

    # 4. Formatting and reference lines
    rate_2_in_50 = -np.log(1 - 0.02) / 50
    # Changed to gray dashed to not clash visually with the 'rocket' palette
    ax1.axvline(x=rate_2_in_50, linestyle="--", c="gray", label="2% in 50yr")
    ax2.axvline(x=rate_2_in_50, linestyle="--", c="gray", label="2% in 50yr")

    if "n" in ds_site.attrs:
        ax2.axhline(
            y=ds_site.attrs["n"], color="black", linestyle=":", label="Baseline N"
        )

    # Ax 1 format
    ax1.set_ylim(1e-2, 100)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ax1.invert_xaxis()
    ax1.set_title(
        r"Relative Empirical Error ($\sqrt{MSE} / \lambda_{total}$ %)", fontsize=13
    )
    ax1.set_ylabel("Error Percentage (%)")
    ax1.set_xlabel(r"Total Hazard ($\lambda$)")
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Ax 2 format
    ax2.set_yscale("log")
    ax2.set_title("Empirical Effective Sample Size (ESS)", fontsize=13)
    ax2.set_ylabel("ESS (Baseline: Equal Allocation)")
    ax2.set_xlabel(r"Total Hazard ($\lambda$)")
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Unified Legend outside the plot
    handles, labels = ax1.get_legend_handles_labels()
    # Filter unique handles/labels to avoid duplicates
    by_label = dict(zip(labels, handles))
    ax2.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        title="Period (s)",
        bbox_to_anchor=(1.25, 1),
    )

    # Title
    site_name = ds_site.site.values
    fig.suptitle(f"Monte Carlo Sampling Efficiency Analysis: {site_name}", fontsize=15)
    fig.tight_layout()

    return fig


def plot_performance_metrics(ds_site):
    # 1. Prepare the Data
    df = ds_site.to_dataframe().reset_index()

    # Calculate Relative RMSE (The 'Error Margin' as a percentage)
    df["rel_rmse_pct"] = (np.sqrt(df["mse"]) / df["hazard"]) * 100
    # 2. Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    # Use a sequential palette for the periods
    palette = sns.color_palette("rocket", n_colors=df["period"].nunique())

    # --- Plot 1: Relative Error ---
    sns.lineplot(
        data=df,
        x="hazard",
        y="rel_rmse_pct",
        hue="period",
        palette=palette,
        ax=ax1,
        legend=False,
        alpha=0.8,
    )
    rate_2_in_50 = -np.log(1 - 0.02) / 50
    ax1.axvline(x=rate_2_in_50, linestyle="-", c="r", label="2% in 50yr")
    ax2.axvline(x=rate_2_in_50, linestyle="-", c="r", label="2% in 50yr")
    if "n" in ds_site.attrs:
        ax2.axhline(y=ds_site.attrs["n"], label="Baseline")
    ax1.set_ylim(1e-2, 100)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()  # High hazard on left, low on right
    ax1.set_title(r"Relative Error ($\sqrt{MSE} / \lambda_{total}$ %)", fontsize=13)
    ax1.set_ylabel("Error Percentage (%)")
    ax1.set_xlabel(r"Total Hazard ($\lambda$)")
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # --- Plot 2: ESS ---
    sns.lineplot(
        data=df,
        x="hazard",
        y="ess",
        hue="period",
        palette=palette,
        ax=ax2,
        alpha=0.8,
    )

    ax2.set_yscale("log")
    ax2.set_title("Effective Sample Size (ESS)", fontsize=13)
    ax2.set_ylabel("ESS (Baseline: Equal Allocation)")
    ax2.set_xlabel(r"Total Hazard ($\lambda$)")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    sns.move_legend(ax2, "upper right")
    fig.suptitle(
        f"Sampling Efficiency Analysis: Site {ds_site.site.values}",
        fontsize=15,
    )
    fig.tight_layout()  # account for suptitle
    return fig


def spatial_error_map(error: xr.Dataset, sites: gpd.GeoDataFrame) -> folium.Map:
    rmse = (np.sqrt(error.mse) / error.hazard).mean("period") * 100
    target_hazard = -np.log(1 - 0.02) / 50
    threshold_vals = (
        np.abs(error.hazard - target_hazard).idxmin("threshold").mean("period")
    )
    rmse_at_target = rmse.sel(threshold=threshold_vals, method="nearest")
    sites["rmse"] = rmse_at_target.to_series()
    return sites.explore("rmse", marker_kwds=dict(radius=10))


def effective_n(density_dataset, m: int, limit=100000):
    arr = density_dataset.values

    result = bisect.bisect_left(
        range(limit), 0, key=lambda n: np.round(n * arr).sum() - m
    )
    return result


@app.command()
def plot_density_inspection(
    density_dataset_path: Path,
    hazard_path: Path,
    ruptures_path: Path,
    site_path: Path,
    n: int,
    target_rate_index: int,
    sites: list[str],
    random: bool = False,
    output: Path | None = None,
) -> None:
    """ """
    if not output:
        print("Must provide output path")
        return
    output.mkdir(parents=True, exist_ok=True)
    ruptures = gpd.read_parquet(ruptures_path).rename_axis("rupture")
    rates = ruptures["rate"].to_xarray()

    with (
        xr.open_dataarray(density_dataset_path, engine="h5netcdf") as density_dataset,
        xr.open_dataset(hazard_path, engine="h5netcdf") as hazard,
    ):
        density_heatmap = plot_kl_divergence_period_heatmap(density_dataset)

        rate_heatmap = plot_pairwise_rate_kl_divergence(density_dataset)

        if not random:
            n = effective_n(density_dataset.isel(period=0, rate=target_rate_index), n)
            total = int(
                np.round(
                    n * density_dataset.isel(period=0, rate=target_rate_index).values
                ).sum()
            )
            print(f"Effective {n=}, with {total=}")

        if random:
            error = multinomial_error(
                density_dataset.isel(period=0, rate=target_rate_index), rates, hazard, n
            )
        else:
            error = mean_squared_error(
                density_dataset.isel(period=0, rate=target_rate_index), rates, hazard, n
            )
    density_heatmap.savefig(output / "period_divergence.png", dpi=300)
    plt.close(density_heatmap)

    rate_heatmap.savefig(output / "rate_divergence.png", dpi=300)
    plt.close(rate_heatmap)
    plot_site_performance_to(error.sel(site=sites), output)
    sites_gdf = gpd.read_parquet(site_path)
    m = spatial_error_map(error, sites_gdf)
    m.save(output / "error_map.html")


def plot_site_performance_to(
    error: xr.Dataset, output: Path, empirical: bool = False
) -> None:
    output_dir = output / "sites"
    output_dir.mkdir(parents=True, exist_ok=True)
    for site in error.site:
        if empirical:
            fig = plot_empirical_performance_metrics(error.sel(site=site))
        else:
            fig = plot_performance_metrics(error.sel(site=site))
        fig.savefig(output_dir / f"{site.item()}.png", dpi=300)
        plt.close(fig)


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
