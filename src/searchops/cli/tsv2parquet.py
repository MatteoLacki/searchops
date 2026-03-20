"""Convert a TSV file to Parquet using DuckDB (streaming, no RAM copy)."""

import argparse
from pathlib import Path
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Convert TSV to Parquet via DuckDB.")
    parser.add_argument("input",  type=Path, help="Input .tsv file")
    parser.add_argument("output", type=Path, help="Output .parquet file")
    args = parser.parse_args()

    duckdb.sql(f"""
        COPY (
            SELECT * FROM read_csv(
                '{args.input}',
                sep='\t',
                quote='',
                ignore_errors=true
            )
        ) TO '{args.output}' (FORMAT PARQUET)
    """)


if __name__ == "__main__":
    main()
