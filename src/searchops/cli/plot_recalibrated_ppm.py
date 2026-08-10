"""Plot the marginal precursor ppm-error distribution before vs. after recalibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchops.recalibration import plot_recalibrated_ppm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay pre- and post-recalibration precursor and fragment "
        "ppm-error distributions from two SAGE search passes."
    )
    parser.add_argument("initial_sage_results_tsv", type=Path, help="results.sage.tsv from the (uncorrected) calibration pass")
    parser.add_argument("sage_results_tsv", type=Path, help="results.sage.tsv from the final (corrected) pass")
    parser.add_argument("initial_matched_fragments", type=Path, help="matched_fragments.sage.tsv from the (uncorrected) calibration pass")
    parser.add_argument("matched_fragments", type=Path, help="matched_fragments.sage.tsv from the final (corrected) pass")
    parser.add_argument("tolerance", type=Path, help="recalibration_tolerance.json (from fit-mz-recalibration)")
    parser.add_argument("plot", type=Path, help="Output diagnostic PNG path")
    parser.add_argument("--fdr", required=True, type=float, help="Peptide-level FDR threshold")
    args = parser.parse_args()

    with args.tolerance.open() as handle:
        tolerance = json.load(handle)

    plot_recalibrated_ppm(
        args.initial_sage_results_tsv, args.sage_results_tsv,
        args.initial_matched_fragments, args.matched_fragments,
        tolerance, args.fdr, args.plot,
    )


if __name__ == "__main__":
    main()
