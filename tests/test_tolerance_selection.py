import numpy as np
import pytest

from searchops.recalibration import (
    _robust_sigma,
    _select_tolerance,
    _symmetric_tolerance,
    _tolerance,
)


def test_robust_sigma_matches_std_on_clean_gaussian():
    rng = np.random.default_rng(0)
    residual = rng.normal(0.0, 4.0, 20_000)
    assert abs(_robust_sigma(residual) - 4.0) <= 0.05 * 4.0


def test_symmetric_tolerance_is_symmetric_around_median_even_when_skewed():
    rng = np.random.default_rng(1)
    residual = np.concatenate([rng.normal(0.0, 2.0, 9_000), rng.normal(15.0, 2.0, 1_000)])
    tol = _symmetric_tolerance(residual, (0.5, 99.5))
    lo, hi = tol["ppm"]
    center = np.median(residual)
    assert abs((hi - center) - (center - lo)) < 1e-9

    emp = _tolerance(residual, (0.5, 99.5))
    emp_lo, emp_hi = emp["ppm"]
    assert abs((emp_hi - center) - (center - emp_lo)) > 1.0


def test_select_tolerance_default_is_theoretic():
    rng = np.random.default_rng(2)
    residual = rng.normal(0.0, 3.0, 5_000)
    mz_config = {"tolerance_percentiles": [0.5, 99.5]}
    assert _select_tolerance(residual, mz_config) == _symmetric_tolerance(residual, (0.5, 99.5))


def test_select_tolerance_empiric():
    rng = np.random.default_rng(3)
    residual = rng.normal(0.0, 3.0, 5_000)
    mz_config = {"tolerance_percentiles": [0.5, 99.5], "tolerance_method": "empiric"}
    assert _select_tolerance(residual, mz_config) == _tolerance(residual, (0.5, 99.5))


def test_select_tolerance_rejects_unknown_method():
    mz_config = {"tolerance_percentiles": [0.5, 99.5], "tolerance_method": "bogus"}
    with pytest.raises(ValueError, match="unknown tolerance method"):
        _select_tolerance(np.array([1.0, 2.0, 3.0]), mz_config)
