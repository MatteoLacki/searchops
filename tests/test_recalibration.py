from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from searchops.recalibration import MzRecalibration, MzRecalibrationDim, recalibrate


# --- MzRecalibrationDim -----------------------------------------------------

def test_dim_round_trip(tmp_path: Path) -> None:
    ppm = np.array([1.0, 2.0, 4.0, 3.0, -1.0], dtype=np.float64)
    dim = MzRecalibrationDim(x_min=100.0, x_max=500.0, ppm=ppm)
    dim.dump(tmp_path / "mz")

    loaded = MzRecalibrationDim.load(tmp_path / "mz")
    assert loaded.x_min == 100.0
    assert loaded.x_max == 500.0
    np.testing.assert_array_equal(loaded.ppm, ppm)


def test_dim_corrector_matches_nodes() -> None:
    ppm = np.array([1.0, 2.0, 4.0, 3.0, -1.0], dtype=np.float64)
    dim = MzRecalibrationDim(x_min=0.0, x_max=4.0, ppm=ppm)
    corrector = dim.corrector()
    for x, expected in zip([0.0, 1.0, 2.0, 3.0, 4.0], ppm):
        assert corrector(x) == pytest.approx(expected)


def test_dim_corrector_interpolates_linearly() -> None:
    ppm = np.array([0.0, 10.0], dtype=np.float64)
    dim = MzRecalibrationDim(x_min=0.0, x_max=1.0, ppm=ppm)
    corrector = dim.corrector()
    assert corrector(0.25) == pytest.approx(2.5)
    assert corrector(0.5) == pytest.approx(5.0)
    assert corrector(0.75) == pytest.approx(7.5)


def test_dim_corrector_clamps_outside_range() -> None:
    ppm = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    dim = MzRecalibrationDim(x_min=10.0, x_max=20.0, ppm=ppm)
    corrector = dim.corrector()
    assert corrector(-100.0) == pytest.approx(1.0)
    assert corrector(1e6) == pytest.approx(3.0)


def test_dim_rejects_too_few_nodes() -> None:
    with pytest.raises(ValueError):
        MzRecalibrationDim(x_min=0.0, x_max=1.0, ppm=np.array([1.0]))


def test_dim_rejects_non_finite_ppm() -> None:
    with pytest.raises(ValueError):
        MzRecalibrationDim(x_min=0.0, x_max=1.0, ppm=np.array([1.0, np.nan]))


def test_dim_rejects_non_increasing_range() -> None:
    with pytest.raises(ValueError):
        MzRecalibrationDim(x_min=1.0, x_max=1.0, ppm=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        MzRecalibrationDim(x_min=1.0, x_max=0.0, ppm=np.array([1.0, 2.0]))


# --- MzRecalibration ---------------------------------------------------------

def test_mz_recalibration_round_trip_multi_dim(tmp_path: Path) -> None:
    mz_dim = MzRecalibrationDim(x_min=100.0, x_max=1000.0, ppm=np.array([1.0, 0.5, -0.5]))
    rt_dim = MzRecalibrationDim(x_min=0.0, x_max=60.0, ppm=np.array([0.1, 0.2, 0.3, 0.4]))
    artifact = MzRecalibration(dims={"mz": mz_dim, "rt": rt_dim}, bias=0.75)

    path = tmp_path / "recal.mzcalib"
    artifact.dump(path)

    loaded = MzRecalibration.load(path)
    assert loaded.bias == pytest.approx(0.75)
    assert set(loaded.dims) == {"mz", "rt"}
    np.testing.assert_array_equal(loaded.dims["mz"].ppm, mz_dim.ppm)
    np.testing.assert_array_equal(loaded.dims["rt"].ppm, rt_dim.ppm)


def test_mz_recalibration_defaults_bias_to_zero(tmp_path: Path) -> None:
    dim = MzRecalibrationDim(x_min=0.0, x_max=1.0, ppm=np.array([1.0, 2.0]))
    artifact = MzRecalibration(dims={"mz": dim})
    artifact.dump(tmp_path / "recal.mzcalib")

    loaded = MzRecalibration.load(tmp_path / "recal.mzcalib")
    assert loaded.bias == 0.0


def test_mz_recalibration_corrector_dispatches_by_dimension() -> None:
    mz_dim = MzRecalibrationDim(x_min=0.0, x_max=1.0, ppm=np.array([1.0, 2.0]))
    artifact = MzRecalibration(dims={"mz": mz_dim})
    assert artifact.corrector("mz")(1.0) == pytest.approx(2.0)
    with pytest.raises(KeyError):
        artifact.corrector("rt")


def test_mz_recalibration_rejects_empty_dims() -> None:
    with pytest.raises(ValueError):
        MzRecalibration(dims={})


# --- recalibrate() wiring -----------------------------------------------------

def _write_calibration_fixtures(root: Path) -> tuple[Path, Path]:
    """A tiny synthetic calibration dataset: every observed ppm error is a
    constant +5.0, so `global_median` recovers it exactly and the resulting
    correction is trivial to check by hand.
    """
    n = 20
    rng = np.random.default_rng(0)
    charge = np.full(n, 2)
    calc_mz = rng.uniform(400.0, 1200.0, size=n)
    true_ppm = 5.0
    exp_mz = calc_mz * (1.0 + true_ppm * 1e-6)
    expmass = (exp_mz - 1.00727646688) * charge

    sage_results = pd.DataFrame({
        "psm_id": np.arange(n),
        "rank": 1,
        "peptide_q": 0.0,
        "label": 1,
        "charge": charge,
        "expmass": expmass,
        "precursor_ppm": true_ppm,
    })
    sage_results_tsv = root / "results.sage.tsv"
    sage_results.to_csv(sage_results_tsv, sep="\t", index=False)

    frag_calc = rng.uniform(100.0, 1500.0, size=n * 3)
    frag_exp = frag_calc * (1.0 + true_ppm * 1e-6)
    matched_fragments = pd.DataFrame({
        "psm_id": np.repeat(np.arange(n), 3),
        "fragment_mz_experimental": frag_exp,
        "fragment_mz_calculated": frag_calc,
    })
    matched_fragments_tsv = root / "matched_fragments.sage.tsv"
    matched_fragments.to_csv(matched_fragments_tsv, sep="\t", index=False)

    return sage_results_tsv, matched_fragments_tsv


def test_recalibrate_mz_recalibration_path_is_additive(tmp_path: Path) -> None:
    sage_results_tsv, matched_fragments_tsv = _write_calibration_fixtures(tmp_path)
    tof2mz = np.linspace(100.0, 1500.0, 50).astype(np.float32)
    config = {"model": "global_median", "tolerance_percentiles": [5, 95], "numba_grid_points": 64}

    baseline_tof2mz, baseline_tolerance = recalibrate(
        sage_results_tsv, matched_fragments_tsv, tof2mz, config, fdr=0.01,
    )

    artifact_path = tmp_path / "recal.mzcalib"
    with_artifact_tof2mz, with_artifact_tolerance = recalibrate(
        sage_results_tsv, matched_fragments_tsv, tof2mz, config, fdr=0.01,
        mz_recalibration_path=artifact_path,
    )

    np.testing.assert_array_equal(baseline_tof2mz, with_artifact_tof2mz)
    assert baseline_tolerance == with_artifact_tolerance

    artifact = MzRecalibration.load(artifact_path)
    assert artifact.bias == 0.0
    mz_corrector = artifact.corrector("mz")

    # Synthetic fixture has an exact, noiseless +5.0 ppm offset, so the fitted
    # correction should recover it almost exactly at every grid point.
    actual_ppm = np.array([mz_corrector(float(x)) for x in tof2mz])
    np.testing.assert_allclose(actual_ppm, 5.0, atol=1e-2)

    # Applying the artifact's own correction in float64 should reproduce
    # `new_tof2mz` up to the float32 downcast `recalibrate()` applies at the end.
    reapplied = tof2mz.astype(np.float64) / (1.0 + actual_ppm * 1e-6)
    np.testing.assert_allclose(reapplied.astype(np.float32), with_artifact_tof2mz, atol=1e-2)
