"""Tests for the single-site single-im experiment"""


import numpy as np
import pytest
from single_site_single_im import psha
from single_site_single_im.psha import SourceModel


def test_rupture_hazard_monotonically_decreasing() -> None:
    """Asserts that hazard decreases monotonically with threshold."""
    thresholds = np.linspace(0.01, 2.0, 10)
    source_model = SourceModel(
        rates=np.array([0.1]), log_means=np.array([0]), log_stds=np.array([0.5])
    )
    results = psha.analytical_hazard(source_model, thresholds)

    # Check that hazard decreases as threshold increases
    assert np.all(np.diff(results) <= 0)


def test_rupture_hazard_mean_rate() -> None:
    """Asserts that hazard at the mean im value is 0.5."""
    threshold = 1.0
    source_model = SourceModel(
        rates=np.array([1.0]),
        log_means=np.array([np.log(threshold)]),
        log_stds=np.array([0.5]),
    )
    result = psha.analytical_hazard(source_model, np.array([threshold]))

    assert result.squeeze().item() == pytest.approx(0.5)


def test_rupture_hazard_approaches_rupture_rate() -> None:
    """Test rupture rate bounds hazard as threshold -> 0."""
    threshold = 1e-5
    rate = 1.0
    source_model = SourceModel(
        rates=np.array([rate]),
        log_means=np.array([-0.5]),
        log_stds=np.array([0.5]),
    )
    result = psha.analytical_hazard(source_model, np.array([threshold]))

    assert result == pytest.approx(rate)


def test_rupture_hazard_approaches_zero_for_large_threshold() -> None:
    """Test hazard approaches 0 as threshold -> infty."""
    threshold = 1e10
    source_model = SourceModel(
        rates=np.array([1.0]),
        log_means=np.array([-0.5]),
        log_stds=np.array([0.5]),
    )
    result = psha.analytical_hazard(source_model, np.array([threshold]))

    assert result == pytest.approx(0)


def test_rupture_hazard_with_zero_rate_is_zero() -> None:
    """Test hazard is 0 if rate is 0."""
    threshold = 0.5
    source_model = SourceModel(
        rates=np.array([0.0]),
        log_means=np.array([-0.5]),
        log_stds=np.array([0.5]),
    )
    result = psha.analytical_hazard(source_model, np.array([threshold]))

    assert result == pytest.approx(0)


def test_rupture_hazard_shape():
    """Checks that rupture hazard will broadcast properly into the expected shape."""
    n_ruptures = 10
    n_thresholds = 50

    # Create dummy inputs
    rates = np.random.rand(n_ruptures)
    means = np.random.rand(n_ruptures)
    stddevs = np.random.rand(n_ruptures)
    thresholds = np.geomspace(1e-3, 2, num=n_thresholds)
    source_model = SourceModel(rates=rates, log_means=means, log_stds=stddevs)

    result = psha.analytical_hazard(source_model, thresholds)

    # Assertions
    assert result.shape == (n_ruptures, n_thresholds), (
        f"Expected shape ({n_ruptures}, {n_thresholds}), but got {result.shape}"
    )

    # Verify that the resulting hazard values are non-negative (basic sanity check)
    assert np.all(result >= 0)


def test_rupture_hazard_threshold_order():
    """Checks that rupture hazard decreases monotonically along the threshold axis."""
    n_ruptures = 10
    n_thresholds = 50

    # Using fixed rates to ensure we don't hit edge cases of zero probability
    # that might make the test pass accidentally.
    rates = np.full((n_ruptures,), 1.0)
    means = np.full((n_ruptures,), 0.5)
    stddevs = np.full((n_ruptures,), 0.2)
    thresholds = np.geomspace(1e-3, 2, num=n_thresholds)
    source_model = SourceModel(rates=rates, log_means=means, log_stds=stddevs)
    result = psha.analytical_hazard(source_model, thresholds)

    # Check monotonicity along axis 1 (the threshold axis)
    diffs = np.diff(result, axis=1)
    assert np.all(diffs <= 0), (
        "Hazard does not monotonically decrease with increasing thresholds. "
        f"Found {np.sum(diffs > 0)} violations."
    )
