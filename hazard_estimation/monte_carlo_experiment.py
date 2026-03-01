"""Multi-site probabilistic seismic hazard analysis tool.

This tool generates bootstrap results for multiple sites and strategies,
storing them in an xarray Dataset, and provides plotting commands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cyclopts.types import ExistingFile
from tqdm.contrib.concurrent import process_map

# Assumes the refactored psha.py is available
from hazard_estimation import psha, strategies

if TYPE_CHECKING:
    from hazard_estimation.psha import BootstrapResult
    from hazard_estimation.strategies import Config, Strategy

app = cyclopts.App(help=__doc__)

# Constants for return periods
TARGETS = {"2% in 50yr": -np.log(1 - 0.02) / 50, "10% in 50yr": -np.log(1 - 0.10) / 50}


@dataclass(frozen=True)
class WorkItem:
    """A single (site, strategy) pair to process."""

    site: str
    strategy_idx: int
    strategy: Strategy
    ruptures: pd.DataFrame
    thresholds: np.ndarray
    n_resamples: int
    seed: int


@dataclass
class ProcessingResult:
    """Result from processing a single work item."""

    site: str
    strategy_idx: int
    strategy_name: str
    result: BootstrapResult


def compute_source_model(ruptures: pd.DataFrame) -> psha.SourceModel:
    """Create SourceModel from rupture data.

    Args:
        ruptures: DataFrame with rate, PGA_mean, PGA_std_Total columns

    Returns:
        SourceModel instance
    """
    return psha.SourceModel(
        rates=ruptures["rate"].to_numpy(),
        log_means=ruptures["PGA_mean"].to_numpy(),
        log_stds=ruptures["PGA_std_Total"].to_numpy(),
    )


def compute_analytical_hazard_curves(
    ruptures: pd.DataFrame, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute analytical hazard curves for all sites.

    Args:
        ruptures: Multi-indexed DataFrame (site, rupture_id)
        thresholds: Array of intensity measure thresholds

    Returns:
        Dictionary mapping site names to hazard curves
    """
    sites = ruptures.index.get_level_values("site").unique()
    curves = {}

    for site in sites:
        site_ruptures = ruptures.loc[site]
        source_model = compute_source_model(site_ruptures)
        curves[site] = psha.analytical_hazard(source_model, thresholds).sum(axis=0)

    return curves


def process_work_item(item: WorkItem) -> ProcessingResult:
    """Process a single (site, strategy) pair.

    This function is designed to be executed in a separate process.

    Args:
        item: WorkItem containing all necessary data

    Returns:
        ProcessingResult with bootstrap results
    """
    # Create local RNG with deterministic seed
    rng = np.random.default_rng(seed=item.seed)

    # Create source model for this site
    source_model = compute_source_model(item.ruptures)

    # Define simulation function that captures the strategy
    def simulation_fn() -> np.ndarray:
        plan = item.strategy(item.ruptures, rng)
        return psha.monte_carlo_rupture_hazard(source_model, plan, item.thresholds, rng)

    # Run bootstrap (without inner progress bar)
    result = psha.run_bootstrap(
        simulation_fn,
        n_resamples=item.n_resamples,
        use_tqdm=False,
    )

    return ProcessingResult(
        site=item.site,
        strategy_idx=item.strategy_idx,
        strategy_name=item.strategy.name,
        result=result,
    )


def create_work_items(
    ruptures: pd.DataFrame,
    config: Config,
    thresholds: np.ndarray,
) -> list[WorkItem]:
    """Create work items for all (site, strategy) pairs.

    Args:
        ruptures: Multi-indexed rupture DataFrame
        config: Strategy configuration
        thresholds: IM thresholds

    Returns:
        List of WorkItem instances
    """
    sites = ruptures.index.get_level_values("site").unique().to_list()
    base_seed = config.seed

    items = []
    for site_idx, site in enumerate(sites):
        site_ruptures = ruptures.loc[site]
        for strat_idx, strategy in enumerate(config.strategies):
            # Deterministic seed for each (site, strategy) pair
            seed = base_seed + site_idx * 1000 + strat_idx
            items.append(
                WorkItem(
                    site=site,
                    strategy_idx=strat_idx,
                    strategy=strategy,
                    ruptures=site_ruptures,
                    thresholds=thresholds,
                    n_resamples=config.n_resamples,
                    seed=seed,
                )
            )

    return items


def build_xarray_dataset(
    results: list[ProcessingResult],
    analytical_curves: dict[str, np.ndarray],
    thresholds: np.ndarray,
    n_resamples: int,
    strategy_names: list[str],
    strategy_colors: list[str],
) -> xr.Dataset:
    """Build xarray Dataset from processing results.

    Args:
        results: List of ProcessingResult instances
        analytical_curves: True hazard curves for each site
        thresholds: IM thresholds
        n_resamples: Number of bootstrap resamples
        strategy_names: Ordered list of strategy names
        strategy_colors: Ordered list of strategy colors

    Returns:
        xarray Dataset with all results
    """
    # Extract dimensions
    sites = sorted(analytical_curves.keys())
    n_sites = len(sites)
    n_strategies = len(strategy_names)
    n_thresholds = len(thresholds)

    # Initialize arrays
    samples = np.zeros((n_strategies, n_sites, n_resamples, n_thresholds))
    mean = np.zeros((n_strategies, n_sites, n_thresholds))
    ci_low = np.zeros((n_strategies, n_sites, n_thresholds))
    ci_high = np.zeros((n_strategies, n_sites, n_thresholds))
    true_hazard = np.zeros((n_sites, n_thresholds))

    # Fill bootstrap results
    for res in results:
        site_idx = sites.index(res.site)
        strat_idx = res.strategy_idx
        samples[strat_idx, site_idx, :, :] = res.result.samples
        mean[strat_idx, site_idx, :] = res.result.mean
        ci_low[strat_idx, site_idx, :] = res.result.ci_low
        ci_high[strat_idx, site_idx, :] = res.result.ci_high

    # Fill analytical hazard
    for site_idx, site in enumerate(sites):
        true_hazard[site_idx, :] = analytical_curves[site]

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "samples": (["strategy", "site", "sample", "threshold"], samples),
            "mean": (["strategy", "site", "threshold"], mean),
            "ci_low": (["strategy", "site", "threshold"], ci_low),
            "ci_high": (["strategy", "site", "threshold"], ci_high),
            "true_hazard": (["site", "threshold"], true_hazard),
        },
        coords={
            "strategy": strategy_names,
            "site": sites,
            "threshold": thresholds,
            "sample": np.arange(n_resamples),
        },
        attrs={
            "strategy_colors": strategy_colors,
            "description": "Multi-site PSHA bootstrap results",
        },
    )

    return ds


@app.command
def generate(
    ruptures_path: ExistingFile,
    strategies_path: ExistingFile,
    output_path: Path,
    *,
    n_workers: int | None = None,
    min_threshold: float = 0.1,
    max_threshold: float = 2.0,
    n_thresholds: int = 100,
) -> None:
    """Generate bootstrap results for all site-strategy pairs.

    Args:
        ruptures_path: Path to parquet file with rupture data
        strategies_path: Path to YAML file with strategy configurations
        output_path: Output path for zarr dataset
        n_workers: Number of parallel workers (default: CPU count)
        min_threshold: Minimum IM threshold (default: 0.1)
        max_threshold: Maximum IM threshold (default: 2.0)
        n_thresholds: Number of threshold points (default: 100)
    """
    print(f"Loading data from {ruptures_path}...")
    ruptures = pd.read_parquet(ruptures_path)

    print(f"Loading strategies from {strategies_path}...")
    config = strategies.load_strategies(strategies_path)

    # Define thresholds
    thresholds = np.geomspace(min_threshold, max_threshold, num=n_thresholds)

    print("Computing analytical hazard curves...")
    analytical_curves = compute_analytical_hazard_curves(ruptures, thresholds)

    print("Creating work items...")
    work_items = create_work_items(ruptures, config, thresholds)

    print(f"Processing {len(work_items)} site-strategy pairs...")
    results = process_map(
        process_work_item,
        work_items,
        max_workers=n_workers,
        desc="Processing site-strategy pairs",
        chunksize=1,
    )

    print("Building xarray dataset...")
    strategy_names = [s.name for s in config.strategies]
    strategy_colors = [s.color for s in config.strategies]

    ds = build_xarray_dataset(
        results=results,
        analytical_curves=analytical_curves,
        thresholds=thresholds,
        n_resamples=config.n_resamples,
        strategy_names=strategy_names,
        strategy_colors=strategy_colors,
    )

    ds.to_netcdf(output_path, mode="w", engine="h5netcdf")


plot_app = cyclopts.App(name="plot", help="Plot bootstrap results from dataset")
app.command(plot_app)


def setup_hazard_axes(ax: plt.Axes, title: str | None = None) -> None:
    """Configure axes for hazard curve plots."""
    ax.set_ylim(bottom=1e-7, top=1e-2)
    ax.set_xlim(0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Acceleration (g)")
    ax.set_ylabel("Annual Probability of Exceedance")
    if title:
        ax.set_title(title)


def plot_confidence_band(
    ax: plt.Axes,
    thresholds: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    *,
    color: str,
    label: str,
    alpha: float = 0.3,
) -> None:
    """Plot confidence interval band."""
    ax.fill_between(thresholds, ci_low, ci_high, color=color, label=label, alpha=alpha)


def add_return_period_targets(
    ax: plt.Axes, thresholds: np.ndarray, *, show_labels: bool = False
) -> None:
    """Add horizontal lines for return period targets."""
    for label, rate in TARGETS.items():
        ax.axhline(rate, color="red", linestyle="--", alpha=0.6, linewidth=1)
        if show_labels:
            ax.text(
                thresholds.min(),
                rate,
                f" {label}",
                va="bottom",
                color="red",
                fontsize=8,
            )


def save_or_show(fig: plt.Figure, save_path: Path | None, *, dpi: int = 300) -> None:
    """Save figure to file or show in window."""
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


@plot_app.command(name="pairwise-comparison")
def plot_pairwise_comparison(
    dataset_path: ExistingFile,
    site: str,
    *,
    save: Path | None = None,
    figsize: tuple[float, float] = (15, 15),
    dpi: int = 300,
) -> None:
    """Create pairwise comparison matrix for all strategies at a site.

    Args:
        dataset_path: Path to zarr dataset
        site: Site name (e.g., 'CBGS')
        save: Optional path to save figure (shows in window if not provided)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
    """
    print(f"Loading dataset from {dataset_path}...")
    ds = xr.open_dataset(dataset_path)

    if site not in ds.site.values:
        raise ValueError(
            f"Site '{site}' not found. Available sites: {list(ds.site.values)}"
        )

    # Select data for site
    site_data = ds.sel(site=site)
    strategies = ds.coords["strategy"].values
    thresholds = ds.coords["threshold"].values
    colors = ds.attrs.get("strategy_colors", [f"C{i}" for i in range(len(strategies))])

    n = len(strategies)

    # Create figure
    fig, axes = plt.subplots(
        n,
        n,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    # Handle single strategy case
    if n == 1:
        axes = np.array([[axes]])

    # Plot pairwise comparisons
    for i, strat_row in enumerate(strategies):
        for j, strat_col in enumerate(strategies):
            ax = axes[i, j]

            # Upper triangle: hide
            if j > i:
                ax.axis("off")
                if i == 0:
                    ax.set_title(strat_col)
                continue

            # Setup axes
            if i == 0:
                setup_hazard_axes(ax, title=strat_col)
            else:
                setup_hazard_axes(ax)

            if j == 0:
                ax.set_ylabel(strat_row)

            # Plot row strategy
            plot_confidence_band(
                ax,
                thresholds,
                site_data.sel(strategy=strat_row)["ci_low"].values,
                site_data.sel(strategy=strat_row)["ci_high"].values,
                color=colors[i],
                label=strat_row,
            )

            # Plot column strategy if different from row
            if i != j:
                plot_confidence_band(
                    ax,
                    thresholds,
                    site_data.sel(strategy=strat_col)["ci_low"].values,
                    site_data.sel(strategy=strat_col)["ci_high"].values,
                    color=colors[j],
                    label=strat_col,
                )

            # Plot true hazard
            ax.plot(
                thresholds,
                site_data["true_hazard"].values,
                c="k",
                lw=2,
                label="True Hazard" if (i == 0 and j == 0) else None,
            )

            # Add target lines
            show_labels = i == 0 and j == 0
            add_return_period_targets(ax, thresholds, show_labels=show_labels)

            ax.legend(fontsize=8)

    fig.suptitle(f"Pairwise Comparison — Site: {site}", fontsize=16)
    save_or_show(fig, save, dpi=dpi)


@plot_app.command
def strategy(
    dataset_path: ExistingFile,
    site: str,
    strategy_name: str,
    *,
    save: Path | None = None,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 300,
) -> None:
    """Plot a single strategy at a specific site.

    Args:
        dataset_path: Path to zarr dataset
        site: Site name
        strategy_name: Strategy name
        save: Optional path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
    """
    print(f"Loading dataset from {dataset_path}...")
    ds = xr.open_dataset(dataset_path)

    # Validate inputs
    if site not in ds.site.values:
        raise ValueError(f"Site '{site}' not found. Available: {list(ds.site.values)}")
    if strategy_name not in ds.strategy.values:
        raise ValueError(
            f"Strategy '{strategy_name}' not found. "
            f"Available: {list(ds.strategy.values)}"
        )

    # Select data
    data = ds.sel(site=site, strategy=strategy_name)
    thresholds = ds.coords["threshold"].values
    colors = ds.attrs.get("strategy_colors", [f"C{i}" for i in range(len(ds.strategy))])
    strat_idx = list(ds.strategy.values).index(strategy_name)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    setup_hazard_axes(ax, title=f"{strategy_name} — Site: {site}")

    # Plot strategy
    plot_confidence_band(
        ax,
        thresholds,
        data["ci_low"].values,
        data["ci_high"].values,
        color=colors[strat_idx],
        label=strategy_name,
    )

    # Plot true hazard
    ax.plot(
        thresholds,
        ds.sel(site=site)["true_hazard"].values,
        c="k",
        lw=2,
        label="True Hazard",
    )

    # Add target lines
    add_return_period_targets(ax, thresholds, show_labels=True)

    ax.legend()
    save_or_show(fig, save, dpi=dpi)


@plot_app.command(name="all-sites")
def plot_all_sites(
    dataset_path: ExistingFile,
    strategy_name: str,
    *,
    save: Path | None = None,
    figsize: tuple[float, float] = (20, 15),
    ncols: int = 4,
    dpi: int = 300,
) -> None:
    """Plot a strategy across all sites in a grid.

    Args:
        dataset_path: Path to zarr dataset
        strategy_name: Strategy name
        save: Optional path to save figure
        figsize: Figure size (width, height) in inches
        ncols: Number of columns in grid
        dpi: Resolution for saved figure
    """
    print(f"Loading dataset from {dataset_path}...")
    ds = xr.open_dataset(dataset_path)

    if strategy_name not in ds.strategy.values:
        raise ValueError(
            f"Strategy '{strategy_name}' not found. "
            f"Available: {list(ds.strategy.values)}"
        )

    sites = ds.coords["site"].values
    thresholds = ds.coords["threshold"].values
    colors = ds.attrs.get("strategy_colors", [f"C{i}" for i in range(len(ds.strategy))])
    strat_idx = list(ds.strategy.values).index(strategy_name)

    # Determine grid layout
    n_sites = len(sites)
    ncols = min(ncols, n_sites)
    nrows = (n_sites + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    # Handle single subplot
    if n_sites == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Plot each site
    for idx, site in enumerate(sites):
        ax = axes[idx]
        data = ds.sel(site=site, strategy=strategy_name)

        setup_hazard_axes(ax, title=site)

        plot_confidence_band(
            ax,
            thresholds,
            data["ci_low"].values,
            data["ci_high"].values,
            color=colors[strat_idx],
            label=strategy_name if idx == 0 else None,
        )

        ax.plot(
            thresholds,
            ds.sel(site=site)["true_hazard"].values,
            c="k",
            lw=2,
            label="True Hazard" if idx == 0 else None,
        )

        add_return_period_targets(ax, thresholds, show_labels=(idx == 0))

        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_sites, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Strategy: {strategy_name} — All Sites", fontsize=16)
    save_or_show(fig, save, dpi=dpi)


@plot_app.command
def boxplot(
    dataset_path: ExistingFile,
    site: str,
    *,
    save: Path | None = None,
    figsize: tuple[float, float] = (18, 8),
    dpi: int = 300,
) -> None:
    """Create boxplot comparison for all strategies at a site.

    Compares the distribution of hazard estimates at return period targets.

    Args:
        dataset_path: Path to zarr dataset
        site: Site name
        save: Optional path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
    """
    print(f"Loading dataset from {dataset_path}...")
    ds = xr.open_dataset(dataset_path)

    if site not in ds.site.values:
        raise ValueError(f"Site '{site}' not found. Available: {list(ds.site.values)}")

    # Select data for site
    site_data = ds.sel(site=site)
    strategies_list = ds.coords["strategy"].values
    thresholds = ds.coords["threshold"].values
    colors = ds.attrs.get(
        "strategy_colors", [f"C{i}" for i in range(len(strategies_list))]
    )

    # Create figure
    fig, axes = plt.subplots(1, len(TARGETS), figsize=figsize)
    if len(TARGETS) == 1:
        axes = [axes]

    for ax, (target_name, target_rate) in zip(axes, TARGETS.items()):
        # Find threshold closest to target rate
        true_hazard = site_data["true_hazard"].values
        thresh_idx = np.argmin(np.abs(true_hazard - target_rate))
        thresh_value = thresholds[thresh_idx]

        # Extract samples for each strategy at this threshold
        plot_data = [
            site_data.sel(strategy=strat)["samples"].values[thresh_idx, :]
            for strat in strategies_list
        ]

        # Create boxplot
        bplot = ax.boxplot(
            plot_data,
            tick_labels=strategies_list,
            patch_artist=True,
            whis=[2.5, 97.5],
        )

        # Color boxes
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add true hazard line
        ax.axhline(
            target_rate, color="black", linestyle="-", linewidth=2, label="True Hazard"
        )

        ax.set_title(f"{target_name}\n(Threshold: {thresh_value:.3f}g)", fontsize=12)
        ax.set_ylabel("Annual Exceedance Rate")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(f"Strategy Comparison — Site: {site}", fontsize=14)
    plt.tight_layout()

    save_or_show(fig, save, dpi=dpi)


@plot_app.command(name="site-summary")
def plot_site_summary(
    dataset_path: ExistingFile,
    *,
    save: Path | None = None,
    figsize: tuple[float, float] = (20, 12),
    dpi: int = 300,
) -> None:
    """Create summary plot showing all strategies across all sites.

    Args:
        dataset_path: Path to zarr dataset
        save: Optional path to save figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
    """
    print(f"Loading dataset from {dataset_path}...")
    ds = xr.open_dataset(dataset_path)

    sites = ds.coords["site"].values
    strategies_list = ds.coords["strategy"].values
    thresholds = ds.coords["threshold"].values
    colors = ds.attrs.get(
        "strategy_colors", [f"C{i}" for i in range(len(strategies_list))]
    )

    # Determine grid layout
    n_sites = len(sites)
    ncols = min(4, n_sites)
    nrows = (n_sites + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if n_sites == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Plot each site with all strategies
    for site_idx, site in enumerate(sites):
        ax = axes[site_idx]
        site_data = ds.sel(site=site)

        setup_hazard_axes(ax, title=site)

        # Plot all strategies
        for strat_idx, strat_name in enumerate(strategies_list):
            data = site_data.sel(strategy=strat_name)
            plot_confidence_band(
                ax,
                thresholds,
                data["ci_low"].values,
                data["ci_high"].values,
                color=colors[strat_idx],
                label=strat_name if site_idx == 0 else None,
                alpha=0.25,
            )

        # Plot true hazard
        ax.plot(
            thresholds,
            site_data["true_hazard"].values,
            c="k",
            lw=2,
            label="True Hazard" if site_idx == 0 else None,
        )

        add_return_period_targets(ax, thresholds, show_labels=(site_idx == 0))

        if site_idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_sites, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("All Strategies — All Sites", fontsize=16)
    save_or_show(fig, save, dpi=dpi)


if __name__ == "__main__":
    app()
