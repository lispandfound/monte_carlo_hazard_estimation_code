"""Load sampling strategies from a toml file."""

import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator

from single_site_single_im import psha
from single_site_single_im.psha import SimulationPlan


class Strategy(typing.Protocol):
    """Generic sampling strategy protocol."""

    color: str

    def __call__(
        self, ruptures: pd.DataFrame, rng: Generator
    ) -> SimulationPlan: ...  # numpydoc ignore=GL08

    def label(self, ruptures: pd.DataFrame) -> str: ...  # numpydoc ignore=GL08


@dataclass
class PoissonStrategy:
    """Poisson sampling strategy."""

    color: str
    """Colour of sampling strategy in plots."""
    length: int
    """Length of poisson catalogue (in years)."""

    def __call__(self, ruptures: pd.DataFrame, rng: Generator) -> SimulationPlan:
        """Create the poisson sampling strategy from a set of ruptures.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.
        rng : Generator
            Random number generator to use.

        Returns
        -------
        pd.DataFrame
            Sampling strategy.

        See Also
        --------
        psha.poisson_catalogue_sampling_strategy : The sampling strategy function this calls.
        """
        return psha.poisson_strategy(ruptures, rng, self.length)

    def label(self, ruptures: pd.DataFrame) -> str:
        """Produce a human-readable description of this sampling strategy.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample

        Returns
        -------
        str
            Sampling strategy description.
        """
        mean_ruptures = psha.poisson_mean_ruptures_sampled(ruptures, self.length)

        return f"Poisson strategy (Y = {self.length}, N_mean = {mean_ruptures})"


@dataclass
class FixedEffortPoissonStrategy:
    """Fixed-effort Poisson sampling strategy.

    Samples from the distribution of the Poisson process governing
    earthquake return periods, where a fixed number of observations
    are returned. Effectively, recording earthquakes as they return
    from the Poisson process until we hit ``n`` returns.

    """

    color: str
    """Colour of sampling strategy in plots."""
    n: int
    """Conditioning number (number of simulations) for Poisson process"""

    def __call__(self, ruptures: pd.DataFrame, rng: Generator) -> SimulationPlan:
        """Create the poisson sampling strategy from a set of ruptures.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.
        rng : Generator
            Random number generator to use.

        Returns
        -------
        pd.DataFrame
            Sampling strategy.

        See Also
        --------
        psha.poisson_catalogue_sampling_strategy : The sampling strategy function this calls.
        """
        return psha.fixed_effort_poisson_strategy(ruptures, rng, self.n)

    def label(self, ruptures: pd.DataFrame) -> str:
        """Produce a human-readable description of this sampling strategy.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample

        Returns
        -------
        str
            Sampling strategy description.
        """

        return f"Poisson strategy (N = {self.n})"


@dataclass
class ImportanceSampledStrategy:
    """Importance sampling strategy.

    Samples from an arbitrary multi-nomial distribution Multinomial(n;
    p_i) and weights rupture estimators so result is unbiased. Fixed
    Poisson strategy is simply this with p_i ~ lambda_{rup, i}.

    """

    color: str
    """Colour of sampling strategy in plots."""
    distribution: pd.DataFrame
    """Distribution of ruptures to sample"""
    n: int
    """Conditioning number (number of simulations) for process"""
    title: str
    """Title of sampling strategy."""

    def __call__(self, ruptures: pd.DataFrame, rng: Generator) -> SimulationPlan:
        """Create the poisson sampling strategy from a set of ruptures.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.
        rng : Generator
            Random number generator to use.

        Returns
        -------
        pd.DataFrame
            Sampling strategy.

        See Also
        --------
        psha.importance_sampled_strategy : The sampling strategy function this calls.
        """
        return psha.importance_sampled_strategy(
            ruptures, self.distribution["probability"], rng, self.n
        )

    def label(self, ruptures: pd.DataFrame) -> str:
        """Produce a human-readable description of this sampling strategy.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample

        Returns
        -------
        str
            Sampling strategy description.
        """

        return self.title


@dataclass
class NaiveStrategy:
    """Naive Monte Carlo sampling strategy."""

    color: str
    """Colour of sampling strategy in plots."""
    n: int
    """Number of samples per rupture."""

    def __call__(self, ruptures: pd.DataFrame, rng: Generator) -> SimulationPlan:
        """Create the naive sampling strategy from a set of ruptures.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.
        rng : Generator
            Generator, ignored.

        Returns
        -------
        pd.DataFrame
            Sampling strategy where every rupture is sampled n times.

        See Also
        --------
        psha.naive_monte_carlo_sampling_strategy : The sampling strategy function this calls.
        """
        return psha.naive_strategy(ruptures, self.n)

    def label(self, ruptures: pd.DataFrame) -> str:
        """Produce a human-readable description of this sampling strategy.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.

        Returns
        -------
        str
            Sampling strategy description including total sample count.
        """
        return f"Naive strategy (N = {self.n * len(ruptures)})"


@dataclass
class SCECStrategy:
    """SCEC CyberShake sampling strategy based on fault area."""

    color: str
    """Colour of sampling strategy in plots."""

    def __call__(self, ruptures: pd.DataFrame, rng: Generator) -> SimulationPlan:
        """Create the SCEC sampling strategy from a set of ruptures.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.
        rng : Generator
            Random number generator to use. Ignored.

        Returns
        -------
        pd.DataFrame
            Sampling strategy proportional to rupture area.

        See Also
        --------
        psha.scec_cybershake_sampling_strategy : The sampling strategy function this calls.
        psha.SCEC_SAMPLING_SPACE : The spacing constant used for this strategy.
        """
        return psha.scec_cybershake_strategy(ruptures, psha.SCEC_SAMPLING_SPACE)

    def label(self, ruptures: pd.DataFrame) -> str:
        """Produce a human-readable description of this sampling strategy.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.

        Returns
        -------
        str
            Sampling strategy description including total calculated samples.
        """
        num_samples = ruptures["area"].sum() / (
            psha.SCEC_SAMPLING_SPACE * psha.SCEC_SAMPLING_SPACE
        )
        return f"SCEC strategy (N = {num_samples})"


@dataclass
class CybershakeStrategy:
    """CyberShake NZ sampling strategy based on magnitude clipping."""

    color: str
    """Colour of sampling strategy in plots."""

    def __call__(self, ruptures: pd.DataFrame, rng: Generator) -> SimulationPlan:
        """Create the CyberShake NZ sampling strategy from a set of ruptures.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.
        rng : Generator
            Random number generator to use. Ignored.

        Returns
        -------
        pd.DataFrame
            Sampling strategy proportional to magnitude (clipped).

        See Also
        --------
        psha.cybershake_nz_sampling_strategy : The sampling strategy function this calls.
        """
        return psha.cybershake_nz_strategy(ruptures)

    def label(self, ruptures: pd.DataFrame) -> str:
        """Produce a human-readable description of this sampling strategy.

        Parameters
        ----------
        ruptures : pd.DataFrame
            Ruptures to sample.

        Returns
        -------
        str
            Sampling strategy description including total calculated samples.
        """
        num_samples = np.round(
            np.clip(27 * ruptures["magnitude"] - 148, 14, 68)
        ).astype(np.int64)
        return f"Cybershake NZ strategy (N = {num_samples})"


@dataclass
class Config:
    """Description of the configuration of sampling strategies."""

    strategies: list[Strategy]
    """Strategies to run."""
    n_resamples: int
    """Number of bootstrap resamples to compute confidence intervals from."""
    seed: int


def load_strategies(config_path: Path) -> Config:
    """Load bootstrap sampling strategies from a toml file.

    Parameters
    ----------
    config_path : Path
        Path to toml file.

    Returns
    -------
    Config
        Configuration read from file.

    Raises
    ------
    ValueError
        If an invalid strategy is described in the config.
    """
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    strategies: list[Strategy] = []

    for name, params in data["strategies"].items():
        match params["type"]:
            case "poisson":
                strategy = PoissonStrategy(
                    color=params["color"], length=params["length"]
                )
            case "fixed_effort_poisson":
                strategy = FixedEffortPoissonStrategy(
                    color=params["color"], n=params["n"]
                )
            case "naive":
                strategy = NaiveStrategy(color=params["color"], n=params["n"])
            case "scec":
                strategy = SCECStrategy(color=params["color"])
            case "cybershake_nz":
                strategy = CybershakeStrategy(color=params["color"])
            case "importance_sampled":
                distribution = pd.read_parquet(params["distribution"])
                strategy = ImportanceSampledStrategy(
                    color=params["color"],
                    distribution=distribution,
                    n=params["n"],
                    title=params["title"],
                )
            case type:
                raise ValueError(f"Strategy {type} not recognised.")

        strategies.append(strategy)

    return Config(
        strategies,
        n_resamples=data["bootstrap"]["n_resamples"],
        seed=data["bootstrap"]["seed"],
    )
