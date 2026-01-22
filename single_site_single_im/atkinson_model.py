"""Module to compute ground motion models for simulations"""

from enum import StrEnum
from pathlib import Path

import cyclopts
import pandas as pd

app = cyclopts.App()


class IntensityMeasure(StrEnum):
    """
    Enum representing various Earthquake Intensity Measures (IMs).
    """

    PGA = "PGA"
    """Peak Ground Acceleration: The maximum ground acceleration that occurred during earthquake shaking."""

    PGV = "PGV"
    """Peak Ground Velocity: The maximum ground velocity reached during earthquake shaking."""

    PSA = "pSA"
    """Pseudo-Spectral Acceleration: An approximation of the maximum acceleration of a SDOF system."""

    AI = "AI"
    """Arias Intensity: A measure of the cumulative intensity of an earthquake, derived from the acceleration record."""

    CAV = "CAV"
    """Cumulative Absolute Velocity: The integral of the absolute acceleration over the duration of the motion."""

    DS575 = "Ds575"
    """Significant Duration (5-75%): The time interval over which 5% to 75% of the Arias Intensity is build up."""

    DS595 = "Ds595"
    """Significant Duration (5-95%): The time interval over which 5% to 95% of the Arias Intensity is build up."""


def ground_motion_model_estimates(
    rupture_df: pd.DataFrame, im: IntensityMeasure, periods: list[float] | None = None
) -> pd.DataFrame:
    """Get ground motion model estimates for the given intensity measure using the Atkinson 2022 model.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        Input rupture dataframe containing source and site measurements.
    im : str
        Intensity measure to estimate for
    periods : list[float] | None
        Periods to compute ground motion model for (if using pSA).

    Returns
    -------
    pd.DataFrame
        Rupture dataframe containing mean, stddev for log(im) for each
        rupture.
    """
    import oq_wrapper as oqw

    tect_type = oqw.constants.TectType.ACTIVE_SHALLOW
    gmm_outputs = oqw.run_gmm(
        oqw.constants.GMM.A_22, tect_type, rupture_df, im, periods=periods
    )
    gmm_outputs.index = rupture_df.index
    return rupture_df.join(gmm_outputs)


@app.default
def run_ground_motion_model(
    rupture_df_path: Path,
    im: IntensityMeasure,
    gmm_output_path: Path,
    periods: list[float] | None = None,
) -> None:
    """Run the ground motion model for the rupture inputs.

    Parameters
    ----------
    rupture_df_path : Path
        Rupture input dataframe.
    im : str
        Intensity measure to calculate.
    gmm_output_path : Path
        Output path (parquet)
    periods : list[float] | None
        Periods to compute ground motion model for (if using pSA).
    """
    rupture_df = pd.read_parquet(rupture_df_path)
    gmm_outputs = ground_motion_model_estimates(rupture_df, im, periods)
    gmm_outputs.to_parquet(gmm_output_path)


if __name__ == "__main__":
    app()
