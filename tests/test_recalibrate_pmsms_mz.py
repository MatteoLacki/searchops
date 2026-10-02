from pathlib import Path

import mmappet
import numpy as np
import pandas as pd
import pytest

from searchops.recalibration import recalibrate_pmsms_mz
from timstofu.mzrecalibration import MzRecalibration


def _true_ppm(mz: np.ndarray, rt_minutes: np.ndarray) -> np.ndarray:
    return 1.0 + 0.002 * (mz - 800.0) + 0.5 * (rt_minutes - 4.0)


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    n_psms, n_fragments = 300, 30
    psm_rt = rng.uniform(0.5, 8.0, n_psms)
    results = pd.DataFrame({
        "psm_id": np.arange(n_psms),
        "rank": 1,
        "peptide_q": 0.001,
        "label": 1,
        "expmass": 1000.0,
        "charge": 2,
        "rt": psm_rt,
    })
    results.to_csv(tmp_path / "results.sage.tsv", sep="\t", index=False)

    psm_id = np.repeat(np.arange(n_psms), n_fragments)
    calculated = rng.uniform(320.0, 1480.0, n_psms * n_fragments)
    ppm = _true_ppm(calculated, psm_rt[psm_id]) + rng.normal(0.0, 0.1, calculated.size)
    experimental = calculated * (1.0 + ppm * 1e-6)
    pd.DataFrame({
        "psm_id": psm_id,
        "fragment_mz_calculated": calculated,
        "fragment_mz_experimental": experimental,
        "closest_fragment_mz_calculated": calculated,
        "closest_fragment_mz_experimental": experimental,
    }).to_csv(tmp_path / "matched_fragments.sage.tsv", sep="\t", index=False)

    # The pmsms' m/z spans 300 .. 1500 Da: tofs 10 .. 90 over the table 150 + 15 * tof.
    with mmappet.DatasetWriter(tmp_path / "pmsms.mmappet") as writer:
        writer.append_df(pd.DataFrame({
            "tof": np.array([50, 10, 90, 30], dtype=np.uint32),
            "intensity": np.ones(4, dtype=np.uint32),
        }))
    with mmappet.DatasetWriter(tmp_path / "tof2mz.mmappet") as writer:
        writer.append_df(pd.DataFrame({"mz": (150.0 + 15.0 * np.arange(100)).astype(np.float32)}))

    precursors = pd.DataFrame({
        "precursor_idx": np.arange(3, dtype=np.uint64),
        "rt": np.array([60.0, 240.0, 420.0]),
        "fragment_spectrum_start": np.array([0, 1, 3], dtype=np.uint64),
        "fragment_event_cnt": np.array([1, 2, 1], dtype=np.uint64),
    })
    with mmappet.DatasetWriter(tmp_path / "precursors.mmappet") as writer:
        writer.append_df(precursors)

    return {"dir": tmp_path, "precursors": precursors}


def _run(dir: Path) -> dict:
    config = {
        "fragment_model": {
            "class": "searchops.models.PSplineModel",
            "kwargs": {"bin_width_da": 100.0, "lam1": 1.0, "lam2": 1.0},
        },
        "mz": {"tolerance_percentiles": [1, 99]},
    }
    return recalibrate_pmsms_mz(
        dir / "results.sage.tsv",
        dir / "matched_fragments.sage.tsv",
        dir / "pmsms.mmappet",
        dir / "tof2mz.mmappet",
        dir / "precursors.mmappet",
        config,
        fdr=0.01,
        output_precursors=dir / "shifted_precursors.mmappet",
        mz_recalibration_path=dir / "fragment.mzcalib",
        plot_path=dir / "fit.png",
    )


def test_writes_mz_only_calibration_and_per_precursor_shift(inputs: dict) -> None:
    dir = inputs["dir"]
    tolerance = _run(dir)

    recalibration = MzRecalibration.load(dir / "fragment.mzcalib")
    assert set(recalibration.dims) == {"mz"}
    assert recalibration.bias == 0.0
    f_mz = recalibration.dims["mz"]
    assert (f_mz.x_min, f_mz.x_max) == pytest.approx((300.0, 1500.0), abs=1.0)

    shifted = mmappet.open_dataset(dir / "shifted_precursors.mmappet")
    expected = inputs["precursors"]
    assert list(shifted.columns) == [*expected.columns, "fragment_shift_ppm"]
    pd.testing.assert_frame_equal(shifted[expected.columns], expected)

    mz = np.array([400.0, 800.0, 1200.0])
    for rt_seconds, shift_ppm in zip(expected["rt"], shifted["fragment_shift_ppm"]):
        np.testing.assert_allclose(
            f_mz.evaluate(mz) + shift_ppm, _true_ppm(mz, np.full(3, rt_seconds / 60.0)), atol=0.1
        )

    lo, hi = tolerance["ppm"]
    assert lo == pytest.approx(-hi, abs=0.05)
    assert 0.1 < hi < 0.5


def test_refuses_to_overwrite_output_precursors(inputs: dict) -> None:
    dir = inputs["dir"]
    (dir / "shifted_precursors.mmappet").mkdir()
    with pytest.raises(FileExistsError):
        _run(dir)
