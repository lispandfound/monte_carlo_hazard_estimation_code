import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from hazard_estimation.psha import (
    STDDEV_LOWER,
    STDDEV_UPPER,
    aggregate_analytical_hazard,
    analytical_threshold_occupancy,
    monte_carlo_threshold_occupancy,
)
from hypothesis import given
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
def rupture_dataframe_strategy(draw, n_ruptures, n_counts: int) -> pd.DataFrame:
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
    counts = draw(
        hpy.arrays(
            np.int64,
            (n_ruptures,),
            elements=st.integers(min_value=1, max_value=n_counts),
        )
    )

    return pd.DataFrame(
        {"area": area, "rake": rake, "rate": rates, "counts": counts}
    ).rename_axis("rupture")


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


def test_monte_carlo_is_unbiased():
    """
    Verifies that the Monte Carlo estimator is unbiased by checking if the
    ensemble mean across many repeats converges to the analytical POE.
    """

    means = np.array([0.5, 1.0, 2.0])
    stddevs = np.array([0.4, 0.8, 1.2])
    thresholds = np.array([0.5, 1.0, 2.0])

    rupture_means, rupture_stddevs = np.meshgrid(means, stddevs)
    rupture_means = rupture_means.ravel()
    rupture_stddevs = rupture_stddevs.ravel()
    rupture = np.arange(len(rupture_means))

    n_samples = 200
    alpha = 0.001

    mean_da = xr.DataArray(rupture_means, dims="rupture", coords={"rupture": rupture})
    std_da = xr.DataArray(rupture_stddevs, dims="rupture", coords={"rupture": rupture})
    samples = xr.DataArray(
        np.full(rupture_means.shape, n_samples, dtype=int),
        dims="rupture",
        coords={"rupture": rupture},
    )

    mc_input_ds = xr.Dataset(
        data_vars=dict(log_mean=np.log(mean_da), log_stddev=std_da)
    ).expand_dims(site=[0])
    mc_input_ds["samples"] = samples
    mc_estimates = monte_carlo_threshold_occupancy(mc_input_ds, thresholds)

    analytical_gmm_ds = xr.Dataset(
        data_vars=dict(log_mean=np.log(mean_da), log_stddev=std_da)
    ).expand_dims(z=[0])

    p_analytical = analytical_threshold_occupancy(analytical_gmm_ds, thresholds)

    theoretical_variance = p_analytical * (1 - p_analytical)
    standard_error = np.sqrt(theoretical_variance / n_samples)

    z_score = np.abs(mc_estimates - p_analytical) / np.maximum(standard_error, 1e-9)

    mask = (p_analytical > 1e-5) & (p_analytical < (1.0 - 1e-5))
    is_outlier = z_score > 4.0

    total_valid_tests = mask.sum().item()
    total_failures = is_outlier.where(mask, drop=True).sum().item()

    max_allowed_failures = sp.stats.binom.ppf(0.999, total_valid_tests, alpha)

    assert total_failures <= max_allowed_failures, (
        f"Ensemble mean biased! Found {total_failures} grid points exceeding 4-sigma "
        f"deviation. Max expected: {max_allowed_failures}."
    )


# def test_monte_carlo_variance():
#     """
#     Verifies MC variance across a parameter grid in a single vectorized pass.
#     """
#     means = np.array([0.5, 1.0, 2.0])
#     stddevs = np.array([0.4, 0.8, 1.2])
#     thresholds = np.array([0.5, 1.0, 2.0])

#     rupture_means, rupture_stddevs = np.meshgrid(means, stddevs)
#     rupture_means = rupture_means.ravel()
#     rupture_stddevs = rupture_stddevs.ravel()
#     rupture = np.arange(len(rupture_means))

#     n_repeats = 1000
#     n_samples = 200
#     alpha = 0.001 / len(thresholds)

#     mean_da = xr.DataArray(rupture_means, dims="rupture", coords={"rupture": rupture})
#     std_da = xr.DataArray(rupture_stddevs, dims="rupture", coords={"rupture": rupture})
#     samples = xr.DataArray(
#         np.full(rupture_means.shape, n_samples, dtype=int),
#         dims="rupture",
#         coords={"rupture": rupture},
#     )

#     mc_input_ds = (
#         xr.Dataset(data_vars=dict(log_mean=np.log(mean_da), log_stddev=std_da))
#         .expand_dims(site=[0])
#         .expand_dims(repeat=np.arange(n_repeats))
#     )
#     mc_input_ds["samples"] = samples

#     mc_estimates = monte_carlo_threshold_occupancy(mc_input_ds, thresholds)

#     analytical_gmm_ds = xr.Dataset(
#         data_vars=dict(log_mean=np.log(mean_da), log_stddev=std_da)
#     ).expand_dims(z=[0])

#     p_analytical = analytical_threshold_occupancy(
#         analytical_gmm_ds, thresholds, weights=1.0
#     )

#     sample_var = mc_estimates.var("repeat", ddof=1)
#     theoretical_var = (p_analytical * (1 - p_analytical)) / n_samples

#     df = n_repeats - 1
#     lower_bound = sp.stats.chi2.ppf(alpha / 2, df) * theoretical_var / df
#     upper_bound = sp.stats.chi2.ppf(1 - alpha / 2, df) * theoretical_var / df

#     is_outlier = (sample_var < lower_bound) | (sample_var > upper_bound)

#     mask = (p_analytical > 1e-5) & (p_analytical < (1.0 - 1e-5))
#     total_valid_tests = mask.sum().item()
#     total_failures = is_outlier.where(mask, drop=True).sum().item()

#     max_allowed_failures = sp.stats.binom.ppf(0.99, total_valid_tests, alpha)

#     assert total_failures <= max_allowed_failures, (
#         f"Observed {total_failures} failures in {total_valid_tests} tests. "
#         f"Maximum expected by chance is {max_allowed_failures}."
#     )


# def test_variable_sampling_variance():
#     means = np.array([0.5, 1.0, 2.0])
#     stddevs = np.array([0.4, 0.8, 1.2])
#     thresholds = np.array([0.5, 1.0, 2.0])

#     samples = np.array([100, 200, 300])
#     rupture_means, rupture_stddevs, rupture_samples = np.meshgrid(
#         means, stddevs, samples
#     )
#     rupture_means = rupture_means.ravel()
#     rupture_stddevs = rupture_stddevs.ravel()
#     rupture_samples = rupture_samples.ravel()
#     rupture = np.arange(len(rupture_means))

#     n_repeats = 1000

#     alpha = 0.001 / len(thresholds)
#     mean_da = xr.DataArray(rupture_means, dims="rupture", coords={"rupture": rupture})
#     std_da = xr.DataArray(rupture_stddevs, dims="rupture", coords={"rupture": rupture})
#     samples = xr.DataArray(
#         rupture_samples,
#         dims="rupture",
#         coords={"rupture": rupture},
#     )

#     mc_input_ds = (
#         xr.Dataset(data_vars=dict(log_mean=np.log(mean_da), log_stddev=std_da))
#         .expand_dims(site=[0])
#         .expand_dims(repeat=np.arange(n_repeats))
#     )
#     mc_input_ds["samples"] = samples

#     mc_estimates = monte_carlo_threshold_occupancy(mc_input_ds, thresholds)

#     analytical_gmm_ds = xr.Dataset(
#         data_vars=dict(log_mean=np.log(mean_da), log_stddev=std_da)
#     ).expand_dims(z=[0])

#     p_analytical = analytical_threshold_occupancy(
#         analytical_gmm_ds, thresholds, weights=1.0
#     )

#     sample_var = mc_estimates.var("repeat", ddof=1)
#     theoretical_var = (p_analytical * (1 - p_analytical)) / samples

#     df = n_repeats - 1
#     lower_bound = sp.stats.chi2.ppf(alpha / 2, df) * theoretical_var / df
#     upper_bound = sp.stats.chi2.ppf(1 - alpha / 2, df) * theoretical_var / df

#     is_outlier = (sample_var < lower_bound) | (sample_var > upper_bound)

#     # Because thresholds are nested, we should consider that outliers
#     # are not independent. We will do this by reducing the sensitivity
#     # of the test to consider failures over the threshold space
#     # instead.
#     rupture_is_outlier = is_outlier.any(dim="threshold")

#     # 2. Now your successes and trials are perfectly aligned
#     total_failures = rupture_is_outlier.sum().item()
#     total_valid_tests = len(rupture)  # Number of independent ruptures
#     max_allowed_failures = sp.stats.binom.ppf(0.999, total_valid_tests, alpha)

#     assert total_failures <= max_allowed_failures, (
#         f"Observed {total_failures} failures in {total_valid_tests} tests. "
#         f"Maximum expected by chance is {max_allowed_failures}."
#     )


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
    xr.testing.assert_allclose(
        hazard.sel(threshold=1e-9, method="nearest").drop_vars("threshold"),
        rates.expand_dims(site=hazard.site),
        rtol=0.01,
        check_dim_order=False,
    )
    assert np.all(hazard.diff("threshold") <= 1e-12)


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
