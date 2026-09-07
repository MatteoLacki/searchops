"""Fit a fragment m/z recalibration model and apply it to a pmsms.mmappet's mz column."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from searchops.recalibration import recalibrate_pmsms_mz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit config['fragment_model'] from confident SAGE PSMs and apply "
        "it to an MzPmsms dataset's mz column in one pass."
    )
    parser.add_argument("sage_results_tsv", type=Path, help="results.sage.tsv from the calibration pass")
    parser.add_argument("matched_fragments", type=Path, help="matched_fragments.sage.tsv from the calibration pass")
    parser.add_argument("mz_pmsms", type=Path, help="Input MzPmsms pmsms.mmappet dataset")
    parser.add_argument(
        "precursors", type=Path,
        help="PreSageFilteredPrecursors mmappet dataset (rt/fragment_spectrum_start/"
        "fragment_event_cnt source for the fragment RT-bias term)",
    )
    parser.add_argument("output_pmsms", type=Path, help="Output recalibrated pmsms.mmappet dataset")
    parser.add_argument("mz_recalibration", type=Path, help="Output MzRecalibration grid artifact path (.mzcalib)")
    parser.add_argument("tolerance", type=Path, help="Output fragment tolerance JSON path")
    parser.add_argument("plot", type=Path, help="Output diagnostic fit plot PNG path")
    parser.add_argument("--config", required=True, type=Path, help="Recalibration TOML config")
    parser.add_argument("--fdr", required=True, type=float, help="Peptide-level FDR threshold")
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)

    tolerance = recalibrate_pmsms_mz(
        args.sage_results_tsv, args.matched_fragments, args.mz_pmsms, args.precursors,
        config, args.fdr,
        output_pmsms=args.output_pmsms,
        mz_recalibration_path=args.mz_recalibration,
        plot_path=args.plot,
    )

    args.tolerance.parent.mkdir(parents=True, exist_ok=True)
    with args.tolerance.open("w") as handle:
        json.dump(tolerance, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
