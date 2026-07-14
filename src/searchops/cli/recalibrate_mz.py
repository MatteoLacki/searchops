"""Fit and apply a precursor m/z recalibration; narrow SAGE's search tolerances."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from timstofu.binary.array_serialization import dump_to_folder, load_from_folder

from searchops.recalibration import recalibrate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a precursor m/z recalibration from confident SAGE PSMs and "
        "apply it to a tof2mz array."
    )
    parser.add_argument("sage_results_tsv", type=Path, help="results.sage.tsv from the calibration pass")
    parser.add_argument("tof2mz", type=Path, help="Input tof2mz mmappet array")
    parser.add_argument("recalibrated_tof2mz", type=Path, help="Output corrected tof2mz mmappet array")
    parser.add_argument("tolerance", type=Path, help="Output tolerance JSON path")
    parser.add_argument("--config", required=True, type=Path, help="Recalibration TOML config")
    parser.add_argument("--fdr", required=True, type=float, help="Peptide-level FDR threshold")
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)

    tof2mz = load_from_folder(args.tof2mz)
    new_tof2mz, tolerance = recalibrate(args.sage_results_tsv, tof2mz, config, args.fdr)

    dump_to_folder(new_tof2mz, args.recalibrated_tof2mz)
    args.tolerance.parent.mkdir(parents=True, exist_ok=True)
    with args.tolerance.open("w") as handle:
        json.dump(tolerance, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
