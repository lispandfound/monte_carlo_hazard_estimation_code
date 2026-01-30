from pathlib import Path
from typing import Any

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cyclopts.types import Directory, ExistingFile

# Assumes the refactored psha.py is available
from single_site_single_im import psha, strategies
from single_site_single_im.psha import BootstrapResult
from single_site_single_im.strategies import Strategy

app = cyclopts.App()

# Constants for return periods
TARGETS = {"2% in 50yr": -np.log(1 - 0.02) / 50, "10% in 50yr": -np.log(1 - 0.10) / 50}


def get_data_and_analytical_hazard(ruptures_path: Path):
    ruptures = pd.read_parquet(ruptures_path)
    # Define thresholds once to be used everywhere
    thresholds = np.geomspace(0.1, 2.0, num=100)

    # Create the clean SourceModel (decoupling schema from logic)
    source_model = psha.SourceModel(
        rates=ruptures["rate"].values,
        log_means=ruptures["PGA_mean"].values,
        log_stds=ruptures["PGA_std_Total"].values,
    )

    # Compute analytical hazard using the model
    analytical_curve = psha.analytical_hazard(source_model, thresholds).sum(axis=0)

    # Return a dataframe for the analytical curve for easy plotting compatibility
    analytical_df = pd.DataFrame({"hazard": analytical_curve}, index=thresholds)
    analytical_df.index.name = "threshold"

    return ruptures, source_model, thresholds, analytical_df


def setup_axes(ax: plt.Axes, title: str | None) -> None:
    ax.set_ylim(bottom=1e-7, top=1e-2)
    ax.set_xlim(0.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Acceleration (g)")
    ax.set_ylabel("Annual Probability of Exceedence")
    if title:
        ax.set_title(title)


def plot_strategy(
    ax: plt.Axes, thresholds: np.ndarray, result: BootstrapResult, **kwargs: Any
) -> None:
    # We explicitly pass thresholds now, as BootstrapResult is generic
    ax.fill_between(thresholds, result.ci_low, result.ci_high, **kwargs)


def _plot_hazard_comparison(
    ax: plt.Axes,
    true_hazard: np.ndarray,
    thresholds: np.ndarray,
    strat_row: Strategy,
    res_row: BootstrapResult,
    strat_col: Strategy | None = None,
    res_col: BootstrapResult | None = None,
    rupture_df: pd.DataFrame | None = None,
    show_labels: bool = False,
) -> None:
    """Core plotting utility for comparing one or two strategies against truth."""
    label_row = strat_row.label(rupture_df) if rupture_df is not None else "Row"

    # Plot first strategy
    plot_strategy(
        ax, thresholds, res_row, color=strat_row.color, label=label_row, alpha=0.3
    )

    # Plot second strategy if provided
    if strat_col and res_col:
        label_col = strat_col.label(rupture_df) if rupture_df is not None else "Col"
        plot_strategy(
            ax, thresholds, res_col, color=strat_col.color, label=label_col, alpha=0.3
        )

    # Plot True Hazard
    ax.plot(
        thresholds,
        true_hazard,
        c="k",
        lw=2,
        label="True Hazard" if show_labels else None,
    )

    # Plot Target lines
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


def create_pairwise_matrix(
    rupture_df: pd.DataFrame,
    thresholds: np.ndarray,
    strategies: list[Strategy],
    results: list[BootstrapResult],
    true_hazard: pd.DataFrame,
    subsize: tuple[float, float],
) -> plt.Figure:
    n = len(strategies)
    width, height = subsize
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(width * n, height * n),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if n == 1:
        axes = np.array([[axes]])

    for i, (strat_row, result_row) in enumerate(zip(strategies, results)):
        for j, (strat_col, result_col) in enumerate(zip(strategies, results)):
            ax = axes[i, j]
            col_label = strat_col.label(rupture_df)
            row_label = strat_row.label(rupture_df)

            if j > i:
                ax.axis("off")
                if i == 0:
                    ax.set_title(col_label)
                continue

            # Reuse core plotting logic
            if i == 0:
                setup_axes(ax, title=col_label)
            if j == 0:
                setup_axes(ax, None)
                ax.set_ylabel(row_label)
            if i != 0 or j != 0:
                ax.set_xlabel("")

            _plot_hazard_comparison(
                ax,
                true_hazard,
                thresholds,
                strat_row,
                result_row,
                strat_col if i != j else None,
                result_col,
                rupture_df,
                show_labels=(i == 0 and j == 0),
            )
            ax.legend()

    return fig


def create_summary_boxplots(
    ruptures: pd.DataFrame,
    thresholds: np.ndarray,
    strategies: list[Strategy],
    results: list[BootstrapResult],
    true_hazard: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(24, 12))

    for ax, (title, target_rate) in zip(axes, TARGETS.items()):
        # Find the threshold index corresponding to the target rate
        idx = (true_hazard["hazard"] - target_rate).abs().idxmin()
        thresh_idx = np.argmin(np.abs(thresholds - idx))

        plot_data = [res.samples[thresh_idx, :] for res in results]
        labels = [strat.label(ruptures) for strat in strategies]
        colors = [strat.color for strat in strategies]

        bplot = ax.boxplot(
            plot_data, tick_labels=labels, patch_artist=True, whis=[2.5, 97.5]
        )

        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(target_rate, color="black", linestyle="-", label="True Hazard")
        ax.set_title(f"Hazard Distribution at {title}\n(IML context: {idx:.2f})")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


@app.command
def compare(
    strategies_path: ExistingFile,
    ruptures: ExistingFile,
    output: Directory,
    subwidth: int = 5,
    subheight: int = 5,
    figwidth: int = 10,
    figheight: int = 10,
    dpi: int = 300,
) -> None:
    # 1. Load Data and create the optimized SourceModel
    df, source_model, thresholds, true_haz = get_data_and_analytical_hazard(ruptures)

    config = strategies.load_strategies(strategies_path)
    sample_strategies = config.strategies
    results = []

    # 2. Run Bootstraps
    # We use a distinct RNG for each strategy to ensure reproducibility
    rng = np.random.default_rng(seed=config.seed)

    for strategy in sample_strategies:
        # We create a closure that binds the specific strategy and data
        # to the simulation kernel. The bootstrapper just executes this blind.
        def simulation_fn():
            # Generate the plan (integers and weights)
            plan = strategy(df, rng)
            # Execute the kernel
            return psha.monte_carlo_rupture_hazard(source_model, plan, thresholds, rng)

        results.append(
            psha.run_bootstrap(
                simulation_fn,
                n_resamples=config.n_resamples,
                tqdm_desc=strategy.label(df),
            )
        )

    output.mkdir(parents=True, exist_ok=True)

    # 3. Pairwise Matrix Plot
    fig = create_pairwise_matrix(
        df, thresholds, sample_strategies, results, true_haz, (subwidth, subheight)
    )
    fig.suptitle("Pairwise Comparison of Sampling Strategies", fontsize=16)
    fig.savefig(output / "comparison.png", dpi=dpi)
    plt.close(fig)

    # 4. Summary Boxplot
    boxplot_fig = create_summary_boxplots(
        df, thresholds, sample_strategies, results, true_haz
    )
    boxplot_fig.savefig(output / "boxplot.png", dpi=dpi)
    plt.close(boxplot_fig)

    # 5. Individual Pairwise Plots
    pairs_dir = output / "pairs"
    pairs_dir.mkdir(exist_ok=True)

    for i, (s_row, r_row) in enumerate(zip(sample_strategies, results)):
        for j, (s_col, r_col) in enumerate(zip(sample_strategies, results)):
            if j > i:
                continue

            fig_single, ax_single = plt.subplots(figsize=(figwidth, figheight))
            title = (
                f"{s_row.label(df)} vs {s_col.label(df)}" if i != j else s_row.label(df)
            )

            setup_axes(ax_single, title=title)

            _plot_hazard_comparison(
                ax_single,
                true_haz,
                thresholds,
                s_row,
                r_row,
                s_col if i != j else None,
                r_col,
                df,
                show_labels=True,
            )
            ax_single.legend()

            fig_single.savefig(pairs_dir / f"pair_{i}_{j}.png", dpi=dpi)
            plt.close(fig_single)


if __name__ == "__main__":
    app()
