"""Apply the precursor-fit ppm correction to a precursors mmappet dataset's mz column."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import mmappet

from searchops.recalibration import filter_top_psms, fit_correction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the precursor-fit ppm correction to a precursors "
        "mmappet dataset's mz column."
    )
    parser.add_argument("sage_results_tsv", type=Path, help="results.sage.tsv from the calibration pass")
    parser.add_argument("precursors", type=Path, help="Input precursors mmappet directory")
    parser.add_argument("recalibrated_precursors", type=Path, help="Output corrected precursors mmappet directory")
    parser.add_argument("--config", required=True, type=Path, help="Recalibration TOML config")
    parser.add_argument("--fdr", required=True, type=float, help="Peptide-level FDR threshold")
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)

    df = filter_top_psms(args.sage_results_tsv, args.fdr)
    correction = fit_correction(df, config)

    precursors = mmappet.open_dataset(args.precursors)
    precursors = precursors.rename(columns={"mz": "mz_old"})
    old_mz = precursors["mz_old"].to_numpy()
    precursors["mz"] = old_mz / (1.0 + correction(old_mz) * 1e-6)

    with mmappet.DatasetWriter(args.recalibrated_precursors, overwrite_dir=True) as writer:
        writer.append_df(precursors)


if __name__ == "__main__":
    main()
