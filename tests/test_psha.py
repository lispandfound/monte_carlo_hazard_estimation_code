import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hazard_estimation.psha import (
    STDDEV_LOWER,
    STDDEV_UPPER,
    aggregate_analytical_hazard,
    analytical_threshold_occupancy,
    monte_carlo_threshold_occupancy,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hpy


@st.composite
def thresholds_strategy(draw):
    """Generates a sorted array of positive threshold values."""
    thresholds = draw(
        st.lists(
            st.floats(
                min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False
            ),
            min_size=5,
            max_size=20,
            unique=True,
        )
    )
    return np.array(sorted(thresholds))


@st.composite
def rupture_dataframe_strategy(draw, n_ruptures) -> pd.DataFrame:
    log_area = draw(hpy.arrays(np.float64, (n_ruptures,), elements=st.floats(2.0, 3.5)))
    area = 10**log_area
    rake = draw(
        hpy.arrays(np.float64, (n_ruptures,), elements=st.floats(-180.0, 180.0))
    )
    rates = draw(
        hpy.arrays(
            np.float64,
            (n_ruptures,),
            elements=st.floats(min_value=0.001, max_value=0.5),
        )
    )

    return pd.DataFrame({"area": area, "rake": rake, "rates": rates})


@st.composite
def analytical_gmm_strategy(
    draw, n_sites=2, n_ruptures=3, n_z=10, min_mean: float = -5.0, max_mean: float = 2.0
):
    """Generates a synthetic GMM output dataset for Analytical integration."""
    log_mean = draw(
        st.lists(
            st.floats(min_value=min_mean, max_value=max_mean),
            min_size=n_sites * n_ruptures * n_z,
            max_size=n_sites * n_ruptures * n_z,
        )
    )
    log_stddev = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=1.0),
            min_size=n_sites * n_ruptures * n_z,
            max_size=n_sites * n_ruptures * n_z,
        )
    )

    log_mean_arr = np.array(log_mean).reshape((n_sites, n_ruptures, n_z))
    log_stddev_arr = np.array(log_stddev).reshape((n_sites, n_ruptures, n_z))

    # Standard normal z-scores mapping to the STDDEV limits
    z_arr = np.linspace(STDDEV_LOWER, STDDEV_UPPER, n_z)

    ds = xr.Dataset(
        data_vars={
            "log_mean": (("site", "rupture", "z"), log_mean_arr),
            "log_stddev": (("site", "rupture", "z"), log_stddev_arr),
        },
        coords={
            "site": np.arange(n_sites),
            "rupture": np.arange(n_ruptures),
            "z": z_arr,
        },
    )
    return ds


@st.composite
def rates_strategy(draw, n_ruptures=3):
    """Generates valid Poisson occurrence rates."""
    rates = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=0.5),
            min_size=n_ruptures,
            max_size=n_ruptures,
        )
    )
    return xr.DataArray(
        rates, dims=("rupture",), coords={"rupture": np.arange(n_ruptures)}
    )


# Checks that naive
@given(
    gmm_ds=analytical_gmm_strategy(
        n_sites=1,
        n_ruptures=10,
        # nz = 1 because that way we can resample several times without having to worry about mean weighting.
        n_z=1,
        # Minimum mean of 0.5 to remove rare-event probabilities that are hard to estimate.
        min_mean=0.5,
    ),
    thresholds=thresholds_strategy(),
)
@settings(
    deadline=None
)  # Numba cache needs to be warmed up on the first run so hypothesis will scream about the first run taking too long because it needs to compile. We don't care about that.
def test_monte_carlo_is_unbiased_estimator(gmm_ds, thresholds):
    """Verifies the MC property: that the MC estimate of POE is an unbiased estimate of true hazard."""
    rng = np.random.default_rng(seed=42)

    weights = np.ones_like(gmm_ds.z.values)
    analytical_poe = analytical_threshold_occupancy(gmm_ds, thresholds, weights)

    n_repeats = 500
    output_shape = gmm_ds.log_mean.shape[:-1]  # drop the z axis
    sample_means = np.tile(gmm_ds.log_mean.values.reshape(output_shape), (1, n_repeats))
    sample_std = np.tile(gmm_ds.log_stddev.values.reshape(output_shape), (1, n_repeats))
    ruptures = np.tile(gmm_ds.rupture.values, (n_repeats,))
    mc_gmm_ds = xr.Dataset(
        {
            "log_mean": (("site", "rupture"), sample_means),
            "log_stddev": (("site", "rupture"), sample_std),
        },
        coords=dict(site=gmm_ds.site, rupture=ruptures),
    ).stack(sample=("site", "rupture"))

    mc_poe = monte_carlo_threshold_occupancy(mc_gmm_ds, thresholds, rng)
    theoretical_variance = analytical_poe * (1 - analytical_poe)
    # CLT provides a standard error on the mean estimate that we can use to check tolerance
    standard_error = np.sqrt(theoretical_variance / n_repeats)

    # We use a 4-sigma tolerance.
    # The probability of a false failure at 4-sigma is ~very small
    # If the code falls outside this we should probably consider that the code is wrong
    tolerance = 4 * standard_error

    # Due to floating point error, or just dumb luck we set a minimum tolerance of 10^-5
    tolerance = np.maximum(tolerance, 1e-5)
    meets_tolerance = np.abs(mc_poe - analytical_poe) < tolerance
    assert np.all(meets_tolerance), (
        f"MC converged to {mc_poe} but analytical was {analytical_poe}"
    )


@given(gmm_ds=analytical_gmm_strategy(n_z=10), rates=rates_strategy())
def test_analytical_hazard_properties(gmm_ds, rates):
    """
    Validates:
    1. Hazard approaches the rupture rate as threshold -> 0.
    2. Hazard is strictly monotonically decreasing as threshold increases.
    """
    # Insert an extremely low threshold to test the limit -> 0
    thresholds = np.array([1e-9, 0.5, 1.0, 2.0, 5.0])

    hazard = aggregate_analytical_hazard(gmm_ds, rates, thresholds)
    n_sites = len(gmm_ds.site)
    expected_rates = np.tile(rates.values, (n_sites,)).reshape((n_sites, -1))
    assert hazard.sel(threshold=1e-9, method="nearest").values == pytest.approx(
        expected_rates,
        rel=0.01,
    )
    assert np.all(np.diff(hazard.values, axis=-1) <= 1e-12)


@given(
    gmm_ds=analytical_gmm_strategy(n_sites=3, n_ruptures=4),
    rates=rates_strategy(n_ruptures=4),
    thresholds=thresholds_strategy(),
)
def test_hazard_invariance_to_shuffling(gmm_ds, rates, thresholds):
    """
    Validates that changing the order of ruptures in the inputs does not alter
    the mathematically mapped output for a specific rupture.
    """
    hazard_original = aggregate_analytical_hazard(gmm_ds, rates, thresholds)

    # Shuffle ruptures
    shuffled_ruptures = np.array([3, 0, 2, 1])
    gmm_ds_shuffled = gmm_ds.sel(rupture=shuffled_ruptures)
    rates_shuffled = rates.sel(rupture=shuffled_ruptures)

    hazard_shuffled = aggregate_analytical_hazard(
        gmm_ds_shuffled, rates_shuffled, thresholds
    )

    # Sort them back and compare
    hazard_shuffled_restored = hazard_shuffled.sel(rupture=np.arange(4))

    xr.testing.assert_allclose(hazard_original, hazard_shuffled_restored)


@given(
    gmm_ds=analytical_gmm_strategy(n_sites=1, n_ruptures=1),
    rates=rates_strategy(n_ruptures=1),
    thresholds=thresholds_strategy(),
)
def test_higher_mean_implies_higher_hazard(gmm_ds, rates, thresholds):
    """
    Validates the physical sensitivity property: uniformly increasing the log_mean
    must result in a hazard curve that is greater than or equal to the original.
    """
    hazard_baseline = aggregate_analytical_hazard(gmm_ds, rates, thresholds)

    # Shift log_mean up by a massive factor (+2.0 in log space)
    gmm_ds_shifted = gmm_ds.copy(deep=True)
    gmm_ds_shifted["log_mean"] = gmm_ds_shifted["log_mean"] + 2.0

    hazard_shifted = aggregate_analytical_hazard(gmm_ds_shifted, rates, thresholds)

    # Difference should be >= 0 everywhere
    diff = hazard_shifted.values - hazard_baseline.values

    assert np.all(diff >= -1e-12), "Increased mean resulted in decreased hazard."
