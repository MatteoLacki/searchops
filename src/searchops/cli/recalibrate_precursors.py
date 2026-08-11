"""Fit a precursor m/z recalibration model and apply it to a precursors.mmappet's mz column."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from searchops.recalibration import recalibrate_precursors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit config['precursor_model'] from confident SAGE PSMs and apply "
        "it directly to a precursors dataset's mz column in one pass."
    )
    parser.add_argument("sage_results_tsv", type=Path, help="results.sage.tsv from the calibration pass")
    parser.add_argument("precursors", type=Path, help="Input precursors mmappet directory")
    parser.add_argument("output_precursors", type=Path, help="Output recalibrated precursors mmappet directory")
    parser.add_argument("tolerance", type=Path, help="Output precursor tolerance JSON path")
    parser.add_argument("plot", type=Path, help="Output diagnostic fit plot PNG path")
    parser.add_argument("model", type=Path, help="Output serialized fitted model (inspection only)")
    parser.add_argument("--config", required=True, type=Path, help="Recalibration TOML config")
    parser.add_argument("--fdr", required=True, type=float, help="Peptide-level FDR threshold")
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)

    tolerance = recalibrate_precursors(
        args.sage_results_tsv, args.precursors,
        config, args.fdr,
        output_precursors=args.output_precursors,
        plot_path=args.plot,
        model_path=args.model,
    )

    args.tolerance.parent.mkdir(parents=True, exist_ok=True)
    with args.tolerance.open("w") as handle:
        json.dump(tolerance, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
