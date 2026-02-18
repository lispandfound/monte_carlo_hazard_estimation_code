import functools
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from numpy.random import Generator
from hazard_estimation import psha
from source_modelling import magnitude_scaling

RANDOM_SEED = 1
N_RUPTURES = 10


@pytest.fixture(scope="session")
def rupture_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    mag = rng.uniform(6.0, 8.0, size=N_RUPTURES)
    area = magnitude_scaling.leonard_magnitude_to_area(mag, rake=45.0)  # type: ignore[invalid-argument-type]
    # PGA has some signal + noise. Not meant to be a real GMM. In log-space.
    PGA_mean = -3.75 + (mag - 7) * 0.5 + rng.uniform(low=-1 / mag, high=1 / mag)

    df = pd.DataFrame(
        {
            # Random guessing here.
            "mag": mag,
            "rrup": rng.lognormal(300, np.log(200), size=N_RUPTURES),
            "rate": rng.exponential(8.5e-4, size=N_RUPTURES),
            "area": area,
            "PGA_mean": PGA_mean,
            "PGA_std_Total": 0.6,
        }
    )
    return df


@pytest.mark.slow
@pytest.mark.parametrize(
    ["strategy"],
    [
        (lambda df, rng: psha.naive_strategy(df, n=250),),
        (lambda df, rng: psha.poisson_strategy(df, rng, length=200_000),),
        (lambda df, rng: psha.cybershake_nz_strategy(df),),
        (
            lambda df, rng: psha.scec_cybershake_strategy(
                df, hypocentre_spacing=psha.SCEC_SAMPLING_SPACE
            ),
        ),
        (lambda df, rng: psha.fixed_effort_poisson_strategy(df, rng, n=20_000),),
    ],
    ids=[
        "naive",
        "poisson Y=200k",
        "Cybershake NZ",
        "SCEC Cybershake",
        "Fixed Effort Poisson",
    ],
)
def test_strategy_bias(
    rupture_dataframe: pd.DataFrame,
    strategy: Callable[[pd.DataFrame, Generator], psha.SimulationPlan],
) -> None:
    rng = np.random.default_rng(1)
    thresholds = np.geomspace(0.01, 0.2, num=20)

    source_model = psha.SourceModel(
        rates=rupture_dataframe["rate"].values,
        log_means=rupture_dataframe["PGA_mean"].values,
        log_stds=rupture_dataframe["PGA_std_Total"].values,
    )

    analytical_result = psha.analytical_hazard(source_model, thresholds).sum(axis=0)

    def simulation_fn() -> np.ndarray:
        plan = strategy(rupture_dataframe, rng)
        return psha.monte_carlo_rupture_hazard(source_model, plan, thresholds, rng)

    bootstrap_result = psha.run_bootstrap(
        simulation_fn,
        n_resamples=1000,
        use_tqdm=False,
    )

    assert bootstrap_result.mean == pytest.approx(analytical_result, rel=0.05)
