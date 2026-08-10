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
            "charge": np.array([2, 2], dtype=np.int64),
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
    }


def _run_mapper(paths: dict[str, Path], output: Path) -> None:
    map_sage_to_pmsms(
        filtered_parquet=paths["filtered"],
        matched_fragments=paths["matched"],
        precursors_parquet=paths["precursors"],
        pmsms_dir=paths["pmsms"],
        output=output,
        verbose=False,
    )


def test_mapper_matches_expected_fragments(tmp_path: Path) -> None:
    paths = _write_mapper_inputs(tmp_path)
    out = tmp_path / "out"
    _run_mapper(paths, out)

    mapping = pd.read_parquet(out / "mapping.parquet")
    assert mapping["pmsms_fragment_idx"].tolist() == [0, 2, 3, 5]
    assert mapping["sage_fragment_idx"].tolist() == [0, 1, 2, 3]


def test_missing_mz_fails(tmp_path: Path) -> None:
    paths = _write_mapper_inputs(tmp_path, include_mz=False)

    with pytest.raises(ValueError, match="need an 'mz' column"):
        _run_mapper(paths, tmp_path / "out")
