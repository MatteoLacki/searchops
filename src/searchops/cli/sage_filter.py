"""Filter SAGE results at a given peptide-level FDR threshold and write parquet."""

import argparse
from pathlib import Path

from searchops.recalibration import filter_top_psms


def main():
    parser = argparse.ArgumentParser(
        description="Filter sage results (tsv or parquet) at peptide-level FDR and write filtered parquet."
    )
    parser.add_argument("input", type=Path, help="Input results.sage.tsv (or parquet)")
    parser.add_argument("output", type=Path, help="Output filtered .parquet file")
    parser.add_argument("--fdr", type=float, default=0.01, help="Peptide-level FDR threshold (default: 0.01)")
    args = parser.parse_args()

    df = filter_top_psms(args.input, args.fdr)
    df.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
