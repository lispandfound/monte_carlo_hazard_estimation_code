import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from single_site_single_im import psha, strategies
from single_site_single_im.psha import BootstrapResult
from single_site_single_im.strategies import Strategy

app = cyclopts.App()

# --- Helper Functions ---


def get_data_and_analytical_hazard(ruptures_path: Path):
    """Loads rupture data and calculates the ground truth analytical hazard."""
    ruptures = pd.read_parquet(ruptures_path)
    thresholds = np.geomspace(0.1, 2.0, num=50)

    analytical_hazard = psha.calculate_hazard(
        ruptures,
        thresholds,
        hazard_function=functools.partial(
            psha.analytical_hazard, ruptures["rate"].values.reshape((-1, 1))
        ),
    )
    return ruptures, thresholds, analytical_hazard


def setup_axes(ax: plt.Axes, title: str | None) -> None:
    ax.set_ylim(bottom=1e-7, top=1e-2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Acceleration (g)")
    ax.set_ylabel("Annual Probability of Exceedence")
    if title:
        ax.set_title(title)


def plot_strategy(ax: plt.Axes, result: BootstrapResult, **kwargs: Any) -> None:
    """Executes a bootstrap strategy and overlays its confidence interval."""
    ax.fill_between(result.thresholds, result.ci_low, result.ci_high, **kwargs)


# --- Commands ---


@app.command
def poisson(ruptures: Path, output: Path, length: int = 50000) -> None:
    df, thresholds, true_haz = get_data_and_analytical_hazard(ruptures)
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_axes(ax, f"Poisson Strategy (Y={length})")

    plot_strategy(
        ax,
        lambda: psha.poisson_catalogue_sampling_strategy(df, length),
        df,
        thresholds,
        label="Poisson Estimate",
    )

    ax.plot(true_haz.index, true_haz["hazard"], c="k", label="True Hazard")
    ax.legend()
    fig.savefig(output)


@app.command
def naive(ruptures: Path, output: Path, n: int = 10) -> None:
    df, thresholds, true_haz = get_data_and_analytical_hazard(ruptures)
    fig, ax = plt.subplots(figsize=(10, 10))
    setup_axes(ax, f"Naive Strategy (N={n * len(df)})")

    plot_strategy(
        ax,
        lambda: psha.naive_monte_carlo_sampling_strategy(df, n),
        df,
        thresholds,
        label="Naive Estimate",
    )

    ax.plot(true_haz.index, true_haz["hazard"], c="k", label="True Hazard")
    ax.legend()
    fig.savefig(output)


def create_pairwise_matrix(
    rupture_df: pd.DataFrame,
    strategies: list[Strategy],
    results: list[BootstrapResult],
    true_hazard: pd.DataFrame,
    subsize: tuple[float, float],
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    n = len(strategies)
    (width, height) = subsize
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(width * n, height * n),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    # Handle single strategy case (make it iterable 2D array)
    if n == 1:
        axes = np.array([[axes]])

    for i, (strat_row, result_row) in enumerate(zip(strategies, results)):
        for j, (strat_col, result_col) in enumerate(zip(strategies, results)):
            ax = axes[i, j]
            col_label = strat_col.label(rupture_df)
            row_label = strat_row.label(rupture_df)
            if j > i:
                ax.clear()
                if i == 0:
                    ax.set_title(col_label)

                ax.set_frame_on(False)  # Removes the outer box
                ax.get_xaxis().set_visible(False)  # Hides ticks/labels
                ax.get_yaxis().set_visible(False)  # Hides ticks/labels
                continue

            if i == 0:
                setup_axes(ax, title=col_label)
            if j == 0:
                setup_axes(ax, None)
                ax.set_ylabel(row_label)

            if i != 0 or j != 0:
                ax.set_xlabel("")

            plot_strategy(
                ax, result_row, color=strat_row.color, label=row_label, alpha=0.3
            )

            if i != j:
                plot_strategy(
                    ax, result_col, color=strat_col.color, label=col_label, alpha=0.3
                )

            ax.plot(
                true_hazard.index,
                true_hazard["hazard"],
                c="k",
                linewidth=2,
                label="True Hazard",
            )
            ax.legend()

    return fig, axes


@app.command
def compare(
    strategies_path: Path,
    ruptures: Path,
    output: Path,
    subwidth: int = 4,
    subheight: int = 4,
) -> None:
    df, thresholds, true_haz = get_data_and_analytical_hazard(ruptures)

    config = strategies.load_strategies(strategies_path)
    sample_strategies = config.strategies
    results = []
    for strategy in sample_strategies:
        results.append(
            psha.bootstrap_sampling_strategy(
                strategy,
                df,
                thresholds,
                n_resamples=config.n_resamples,
                desc=strategy.label(df),
            )
        )
    fig, ax = create_pairwise_matrix(
        df, sample_strategies, results, true_haz, subsize=(subwidth, subheight)
    )
    fig.suptitle("Pairwise Comparison of Sampling Strategies", fontsize=16)
    fig.savefig(output)


if __name__ == "__main__":
    app()
