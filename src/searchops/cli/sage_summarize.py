"""Summarize a pre-filtered SAGE parquet into a single-row count TSV."""

import argparse
from pathlib import Path

from searchops.sage import count_sage


def main():
    parser = argparse.ArgumentParser(
        description="Count PSMs/peptides/ions/proteins in a pre-filtered sage parquet."
    )
    parser.add_argument("input",  type=Path, help="Input filtered .parquet file")
    parser.add_argument("output", type=Path, help="Output summary .tsv file")
    args = parser.parse_args()

    counts = count_sage(args.input)
    with open(args.output, "w") as fh:
        fh.write("\t".join(counts.keys()) + "\n")
        fh.write("\t".join(str(v) for v in counts.values()) + "\n")


if __name__ == "__main__":
    main()
