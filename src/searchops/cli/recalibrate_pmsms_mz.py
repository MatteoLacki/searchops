"""Fit a fragment m/z recalibration: an `mz`-only .mzcalib plus each precursor's `fragment_shift_ppm`."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from searchops.recalibration import recalibrate_pmsms_mz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit config['fragment_model'] from confident SAGE PSMs; write f_mz as a .mzcalib "
        "and bias + f_rt(rt) as each precursor's fragment_shift_ppm, for SAGE's --mz-recalibration."
    )
    parser.add_argument("sage_results_tsv", type=Path, help="results.sage.tsv from the calibration pass")
    parser.add_argument("matched_fragments", type=Path, help="matched_fragments.sage.tsv from the calibration pass")
    parser.add_argument("pmsms", type=Path, help="Search pmsms.mmappet dataset (tof column), for the f_mz grid range")
    parser.add_argument("tof2mz_table", type=Path, help="tof -> m/z table mmappet (float32 column mz)")
    parser.add_argument(
        "precursors", type=Path,
        help="PreSageFilteredPrecursors mmappet dataset (raw rt, in seconds)",
    )
    parser.add_argument(
        "output_precursors", type=Path,
        help="Output precursors mmappet: the input plus a fragment_shift_ppm column",
    )
    parser.add_argument("mz_recalibration", type=Path, help="Output MzRecalibration grid artifact path (.mzcalib)")
    parser.add_argument("tolerance", type=Path, help="Output fragment tolerance JSON path")
    parser.add_argument("plot", type=Path, help="Output diagnostic fit plot PNG path")
    parser.add_argument("--config", required=True, type=Path, help="Recalibration TOML config")
    parser.add_argument("--fdr", required=True, type=float, help="Peptide-level FDR threshold")
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)

    tolerance = recalibrate_pmsms_mz(
        args.sage_results_tsv, args.matched_fragments, args.pmsms, args.tof2mz_table, args.precursors,
        config, args.fdr,
        output_precursors=args.output_precursors,
        mz_recalibration_path=args.mz_recalibration,
        plot_path=args.plot,
    )

    args.tolerance.parent.mkdir(parents=True, exist_ok=True)
    with args.tolerance.open("w") as handle:
        json.dump(tolerance, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
