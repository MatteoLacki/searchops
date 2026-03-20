"""Filter a SAGE parquet file at a given peptide-level FDR threshold."""

import argparse
from pathlib import Path
import duckdb


def main():
    parser = argparse.ArgumentParser(
        description="Filter a sage parquet at peptide-level FDR and write filtered parquet."
    )
    parser.add_argument("input",  type=Path, help="Input results.sage.parquet")
    parser.add_argument("output", type=Path, help="Output filtered .parquet file")
    parser.add_argument("--fdr", type=float, default=0.01, help="Peptide-level FDR threshold (default: 0.01)")
    args = parser.parse_args()

    duckdb.sql(f"""
        COPY (
            SELECT * FROM read_parquet('{args.input}')
            WHERE peptide_q <= {args.fdr}
        ) TO '{args.output}' (FORMAT PARQUET)
    """)


if __name__ == "__main__":
    main()
