from pathlib import Path

import mmappet
import numpy as np
import pandas as pd
import pytest

from searchops.cli.sage_pmsms_mapper import map_sage_to_pmsms


FRAG_MZ = np.array(
    [100.125, 200.25, 300.5, 150.125, 250.25, 350.5],
    dtype=np.float32,
)
FRAG_TOF = np.array([0, 2, 4, 1, 3, 5], dtype=np.uint32)
TOF2MZ = np.array([100.125, 150.125, 200.25, 250.25, 300.5, 350.5], dtype=np.float32)
INTENSITY = np.array([100, 200, 300, 400, 500, 600], dtype=np.uint32)
SCORE = np.full(len(FRAG_MZ), 0.5, dtype=np.float32)


def _write_mmappet(path: Path, data: dict[str, np.ndarray]) -> None:
    with mmappet.DatasetWriter(path, overwrite_dir=True) as writer:
        writer.append_df(pd.DataFrame(data))


def _write_mapper_inputs(root: Path, include_mz: bool = True) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)

    pmsms = root / "pmsms.mmappet"
    fragments = {
        "tof": FRAG_TOF,
        "intensity": INTENSITY,
        "score": SCORE,
    }
    if include_mz:
        fragments["mz"] = FRAG_MZ
    _write_mmappet(pmsms, fragments)

    tof2mz = root / "tof2mz.mmappet"
    _write_mmappet(tof2mz, {"x": TOF2MZ})

    precursors = root / "precursors.parquet"
    pd.DataFrame(
        {
            "precursor_idx": np.array([10, 20], dtype=np.int64),
            "fragment_spectrum_start": np.array([0, 3], dtype=np.int64),
            "fragment_event_cnt": np.array([3, 3], dtype=np.int64),
            "charges": np.array([2, 2], dtype=np.int64),
        }
    ).to_parquet(precursors, index=False)

    filtered = root / "filtered.parquet"
    pd.DataFrame(
        {
            "psm_id": ["psm_a", "psm_b"],
            "scannr": ["precursor_idx=10 charge=2", "precursor_idx=20 charge=2"],
        }
    ).to_parquet(filtered, index=False)

    matched = root / "matched.parquet"
    pd.DataFrame(
        {
            "psm_id": ["psm_a", "psm_a", "psm_b", "psm_b"],
            "fragment_mz_experimental": np.array(
                [100.125, 300.5, 150.125, 350.5], dtype=np.float32
            ),
        }
    ).to_parquet(matched, index=False)

    return {
        "filtered": filtered,
        "matched": matched,
        "precursors": precursors,
        "pmsms": pmsms,
        "tof2mz": tof2mz,
    }


def _run_mapper(paths: dict[str, Path], output: Path, tof2mz_path: Path | None) -> None:
    map_sage_to_pmsms(
        filtered_parquet=paths["filtered"],
        matched_fragments=paths["matched"],
        precursors_parquet=paths["precursors"],
        pmsms_dir=paths["pmsms"],
        output=output,
        tof2mz_path=tof2mz_path,
        verbose=False,
    )


def test_tof2mz_mapper_matches_stored_mz_output(tmp_path: Path) -> None:
    legacy = _write_mapper_inputs(tmp_path / "legacy", include_mz=True)
    tof_backed = _write_mapper_inputs(tmp_path / "tof", include_mz=False)

    legacy_out = tmp_path / "legacy_out"
    tof_out = tmp_path / "tof_out"
    _run_mapper(legacy, legacy_out, tof2mz_path=None)
    _run_mapper(tof_backed, tof_out, tof2mz_path=tof_backed["tof2mz"])

    for name in ["precursors.parquet", "mapping.parquet", "mz_delta_quantiles.parquet"]:
        legacy_df = pd.read_parquet(legacy_out / name)
        tof_df = pd.read_parquet(tof_out / name)
        pd.testing.assert_frame_equal(tof_df, legacy_df)

    mapping = pd.read_parquet(tof_out / "mapping.parquet")
    assert mapping["pmsms_fragment_idx"].tolist() == [0, 2, 3, 5]
    assert mapping["sage_fragment_idx"].tolist() == [0, 1, 2, 3]


def test_missing_mz_without_tof2mz_fails(tmp_path: Path) -> None:
    paths = _write_mapper_inputs(tmp_path, include_mz=False)

    with pytest.raises(ValueError, match="unless --tof2mz is provided"):
        _run_mapper(paths, tmp_path / "out", tof2mz_path=None)


def test_tof2mz_mapper_rejects_out_of_bounds_tof(tmp_path: Path) -> None:
    paths = _write_mapper_inputs(tmp_path, include_mz=False)
    short_tof2mz = tmp_path / "short_tof2mz.mmappet"
    _write_mmappet(short_tof2mz, {"x": TOF2MZ[:3]})

    with pytest.raises(ValueError, match="out of bounds"):
        _run_mapper(paths, tmp_path / "out", tof2mz_path=short_tof2mz)


def test_tof2mz_mapper_rejects_non_monotonic_axis(tmp_path: Path) -> None:
    paths = _write_mapper_inputs(tmp_path, include_mz=False)
    bad_tof2mz = tmp_path / "bad_tof2mz.mmappet"
    _write_mmappet(
        bad_tof2mz,
        {"x": np.array([100.125, 90.0, 200.25, 250.25, 300.5, 350.5], dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="non-decreasing"):
        _run_mapper(paths, tmp_path / "out", tof2mz_path=bad_tof2mz)
