import tomllib
import typing
from dataclasses import dataclass

import pandas as pd

from single_site_single_im import psha


class Strategy(typing.Protocol):
    color: str

    def __call__(self, ruptures: pd.DataFrame) -> pd.DataFrame: ...

    def label(self, ruptures: pd.DataFrame) -> str: ...


@dataclass
class PoissonStrategy:
    color: str
    length: int

    def __call__(self, ruptures: pd.DataFrame) -> pd.DataFrame:
        return psha.poisson_catalogue_sampling_strategy(ruptures, self.length)

    def label(self, ruptures: pd.DataFrame) -> str:
        mean_ruptures = psha.poisson_mean_ruptures_sampled(ruptures, self.length)

        return f"Poisson strategy (Y = {self.length}, N_mean = {mean_ruptures})"


@dataclass
class NaiveStrategy:
    color: str
    n: int

    def __call__(self, ruptures: pd.DataFrame) -> pd.DataFrame:
        return psha.naive_monte_carlo_sampling_strategy(ruptures, self.n)

    def label(self, ruptures: pd.DataFrame) -> str:
        return f"Naive strategy (N = {self.n * len(ruptures)})"


@dataclass
class Config:
    strategies: list[Strategy]
    n_resamples: int


def load_strategies(config_path: Path) -> Config:
    # (Pseudocode for loading TOML)
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    strategies: list[Strategy] = []

    for name, params in data["strategies"].items():
        match params["type"]:
            case "poisson":
                strategy = PoissonStrategy(
                    color=params["color"], length=params["length"]
                )
            case "naive":
                strategy = NaiveStrategy(color=params["color"], n=params["n"])
            case type:
                raise ValueError(f"Strategy {type} not recognised.")

        strategies.append(strategy)

    return Config(strategies, n_resamples=data["bootstrap"]["n_resamples"])
