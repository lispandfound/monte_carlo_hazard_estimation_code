"""Tests for the single-site single-im experiment"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from single_site_single_im import psha


def test_rupture_hazard_monotonically_decreasing() -> None:
    """Asserts that hazard decreases monotonically with threshold."""
    thresholds = np.linspace(0.01, 2.0, 10)
    results = psha.analytical_hazard(
        rate=0.1, log_im_mean=0, log_im_stddev=0.5, threshold=thresholds
    )

    # Check that hazard decreases as threshold increases
    assert np.all(np.diff(results) <= 0)


def test_rupture_hazard_mean_rate() -> None:
    """Asserts that hazard at the mean im value is 0.5."""
    threshold = 1.0
    result = psha.analytical_hazard(
        rate=1.0, log_im_mean=np.log(threshold), log_im_stddev=0.5, threshold=threshold
    )

    assert result == pytest.approx(0.5)


def test_rupture_hazard_approaches_rupture_rate() -> None:
    """Test rupture rate bounds hazard as threshold -> 0."""
    threshold = 1e-5
    rate = 1.0
    result = psha.analytical_hazard(
        rate=rate, log_im_mean=-0.5, log_im_stddev=0.5, threshold=threshold
    )

    assert result == pytest.approx(rate)


def test_rupture_hazard_approaches_zero() -> None:
    """Test hazard approaches zero as threshold -> infty."""
    threshold = 1e10
    rate = 1.0
    result = psha.analytical_hazard(
        rate=rate, log_im_mean=-0.5, log_im_stddev=0.5, threshold=threshold
    )

    assert result == pytest.approx(0)


def test_rupture_hazard_zero_rate() -> None:
    """Asserts that hazard with no rate is 0."""
    res = psha.analytical_hazard(rate=0, log_im_mean=1, log_im_stddev=1, threshold=0.5)
    assert res == 0


def test_rupture_hazard_shape():
    """Checks that rupture hazard will broadcast properly into the expected shape."""
    n_ruptures = 10
    n_thresholds = 50

    # Create dummy inputs
    rates = np.random.rand(n_ruptures, 1)
    means = np.random.rand(n_ruptures, 1)
    stddevs = np.random.rand(n_ruptures, 1)
    thresholds = np.geomspace(1e-3, 2, num=n_thresholds)

    # Execute
    result = psha.analytical_hazard(rates, means, stddevs, thresholds)

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
    rates = np.full((n_ruptures, 1), 1.0)
    means = np.full((n_ruptures, 1), 0.5)
    stddevs = np.full((n_ruptures, 1), 0.2)
    thresholds = np.geomspace(1e-3, 2, num=n_thresholds)

    result = psha.analytical_hazard(rates, means, stddevs, thresholds)

    assert result.shape == (n_ruptures, n_thresholds), (
        f"Expected shape ({n_ruptures}, {n_thresholds}), but got {result.shape}"
    )

    # Check monotonicity along axis 1 (the threshold axis)
    diffs = np.diff(result, axis=1)
    assert np.all(diffs <= 0), (
        "Hazard does not monotonically decrease with increasing thresholds. "
        f"Found {np.sum(diffs > 0)} violations."
    )


def test_analytical_psha_calls_hazard_function_correctly():
    """Test that hazard function is called with correct shape and values."""
    # Setup
    rupture_df = pd.DataFrame(
        {"rate": [0.1, 0.2], "PGA_mean": [-1.0, -0.5], "PGA_std_Total": [0.5, 0.6]}
    )
    threshold_values = np.array([0.01, 0.1, 1.0])

    # Mock hazard function
    mock_hazard = Mock(return_value=np.array([[0.05, 0.03, 0.01], [0.10, 0.07, 0.02]]))

    # Execute
    _ = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    # Assert: hazard function called once
    assert mock_hazard.call_count == 1

    # Check arguments passed to hazard function
    call_args = mock_hazard.call_args[0]
    rates, means, stddevs, thresholds = call_args

    # Check shapes: (N_rup, 1) for rates, means, stddevs
    assert rates.shape == (2, 1)
    assert means.shape == (2, 1)
    assert stddevs.shape == (2, 1)
    assert thresholds.shape == (3,)

    # Check values
    assert rates.flatten() == pytest.approx([0.1, 0.2])
    assert means.flatten() == pytest.approx([-1.0, -0.5])
    assert stddevs.flatten() == pytest.approx([0.5, 0.6])
    assert thresholds == pytest.approx(threshold_values)


def test_analytical_psha_aggregates_hazard_correctly():
    """Test that hazard values are summed across ruptures."""
    rupture_df = pd.DataFrame(
        {"rate": [0.1, 0.2], "PGA_mean": [-1.0, -0.5], "PGA_std_Total": [0.5, 0.6]}
    )
    threshold_values = np.array([0.01, 0.1, 1.0])

    # Mock returns (N_rup, N_thresh) array
    mock_hazard = Mock(return_value=np.array([[0.05, 0.03, 0.01], [0.10, 0.07, 0.02]]))

    result = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    # Expected: sum along axis 0 -> [0.15, 0.10, 0.03]
    expected_hazard = np.array([0.15, 0.10, 0.03])
    assert result["hazard"].values == pytest.approx(expected_hazard)


def test_analytical_psha_returns_correct_structure():
    """Test output dataframe structure."""
    rupture_df = pd.DataFrame(
        {"rate": [0.1], "PGA_mean": [-1.0], "PGA_std_Total": [0.5]}
    )
    threshold_values = np.array([0.01, 0.1, 1.0])

    mock_hazard = Mock(return_value=np.array([[0.05, 0.03, 0.01]]))

    result = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "threshold"
    assert "hazard" in result.columns
    assert len(result.columns) == 1
    assert len(result) == len(threshold_values)

    # Check index values
    assert result.index.values == pytest.approx(threshold_values)


def test_analytical_psha_custom_column_names():
    """Test that custom column names are used correctly."""
    rupture_df = pd.DataFrame(
        {"annual_rate": [0.1, 0.2], "mu_log": [-1.0, -0.5], "sigma_log": [0.5, 0.6]}
    )
    threshold_values = np.array([0.01, 0.1])

    mock_hazard = Mock(return_value=np.array([[0.05, 0.03], [0.10, 0.07]]))

    _ = psha.calculate_hazard(
        rupture_df,
        threshold_values,
        mock_hazard,
        rate_col="annual_rate",
        mean_col="mu_log",
        stddev_col="sigma_log",
    )

    # Verify correct columns were extracted
    call_args = mock_hazard.call_args[0]
    rates, means, stddevs, _ = call_args

    assert rates.flatten() == pytest.approx([0.1, 0.2])
    assert means.flatten() == pytest.approx([-1.0, -0.5])
    assert stddevs.flatten() == pytest.approx([0.5, 0.6])


def test_analytical_psha_missing_column_raises_error():
    """Test that missing required columns raise KeyError."""
    rupture_df = pd.DataFrame(
        {
            "rate": [0.1],
            "PGA_mean": [-1.0],
            # Missing PGA_std_Total
        }
    )
    threshold_values = np.array([0.01])
    mock_hazard = Mock()

    with pytest.raises(KeyError):
        psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)


def test_analytical_psha_empty_dataframe():
    """Test with empty rupture dataframe."""
    rupture_df = pd.DataFrame({"rate": [], "PGA_mean": [], "PGA_std_Total": []})
    threshold_values = np.array([0.01, 0.1, 1.0])

    # Mock returns empty array (0, 3)
    mock_hazard = Mock(return_value=np.empty((0, 3)))

    result = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    # Should call hazard function with empty arrays
    call_args = mock_hazard.call_args[0]
    assert call_args[0].shape == (0, 1)  # rates

    # Result should have zeros
    assert len(result) == 3
    assert result["hazard"].values == pytest.approx([0, 0, 0])


def test_analytical_psha_single_threshold():
    """Test with a single threshold value."""
    rupture_df = pd.DataFrame(
        {"rate": [0.1, 0.2], "PGA_mean": [-1.0, -0.5], "PGA_std_Total": [0.5, 0.6]}
    )
    threshold_values = np.array([0.5])

    mock_hazard = Mock(return_value=np.array([[0.05], [0.10]]))

    result = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    assert len(result) == 1
    assert result["hazard"].iloc[0] == pytest.approx(0.15)


def test_analytical_psha_preserves_threshold_order():
    """Test that threshold order is preserved in output."""
    rupture_df = pd.DataFrame(
        {"rate": [0.1], "PGA_mean": [-1.0], "PGA_std_Total": [0.5]}
    )
    # Non-monotonic thresholds
    threshold_values = np.array([1.0, 0.01, 0.5, 0.1])

    mock_hazard = Mock(return_value=np.array([[0.01, 0.09, 0.04, 0.07]]))

    result = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    # Output should maintain order
    assert result.index.values == pytest.approx(threshold_values)
    assert result["hazard"].values == pytest.approx([0.01, 0.09, 0.04, 0.07])


def test_analytical_psha_hazard_function_exception_propagates():
    """Test that exceptions from hazard function propagate."""
    rupture_df = pd.DataFrame(
        {"rate": [0.1], "PGA_mean": [-1.0], "PGA_std_Total": [0.5]}
    )
    threshold_values = np.array([0.01])

    mock_hazard = Mock(side_effect=ValueError("Invalid input"))

    with pytest.raises(ValueError, match="Invalid input"):
        psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)


def test_analytical_psha_with_large_dataset():
    """Test with larger dataset to verify aggregation."""
    n_ruptures = 100
    rupture_df = pd.DataFrame(
        {
            "rate": np.random.uniform(0.01, 0.1, n_ruptures),
            "PGA_mean": np.random.uniform(-2, 0, n_ruptures),
            "PGA_std_Total": np.random.uniform(0.3, 0.8, n_ruptures),
        }
    )
    threshold_values = np.linspace(0.01, 2.0, 20)

    # Mock returns random hazard matrix
    mock_hazard_matrix = np.random.uniform(0, 0.1, (n_ruptures, 20))
    mock_hazard = Mock(return_value=mock_hazard_matrix)

    result = psha.calculate_hazard(rupture_df, threshold_values, mock_hazard)

    # Check that hazard is correctly summed
    expected_hazard = mock_hazard_matrix.sum(axis=0)
    assert result["hazard"].values == pytest.approx(expected_hazard)

    # Verify shape of inputs to hazard function
    call_args = mock_hazard.call_args[0]
    assert call_args[0].shape == (n_ruptures, 1)  # rates
    assert call_args[1].shape == (n_ruptures, 1)  # means
    assert call_args[2].shape == (n_ruptures, 1)  # stddevs
