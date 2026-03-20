"""Write a TSV summary of one or more SAGE result files."""

import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from searchops.sage import summarize_sage

COLUMNS = ["label", "psm_count", "peptide_count", "ion_count", "protein_count"]


def write_summary(
    inputs: list[str | Path],
    output: str | Path | None = None,
    fdr: float = 0.01,
    level: Literal["psm", "peptide", "protein"] = "peptide",
) -> Path | None:
    inputs = [Path(p) for p in inputs]
    rows = []
    for p in tqdm(inputs, unit="file"):
        counts = summarize_sage(p, fdr=fdr, level=level)
        rows.append({"label": str(p), **counts})

    df = pd.DataFrame(rows, columns=COLUMNS)

    if output is None:
        df.to_csv(sys.stdout, sep="\t", index=False)
        return None
    if len(inputs) == 1 and output == "next-to-input":
        output = inputs[0].with_name(inputs[0].stem + ".summary.tsv")
    output = Path(output)
    df.to_csv(output, sep="\t", index=False)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SAGE result file(s) into a TSV."
    )
    parser.add_argument("inputs", nargs="+", help="Input results.sage.tsv/.parquet file(s)")
    parser.add_argument("--output", help="Output TSV path (default: stdout)")
    parser.add_argument("--fdr", type=float, default=0.01, help="FDR threshold (default: 0.01)")
    parser.add_argument(
        "--level",
        choices=["psm", "peptide", "protein"],
        default="peptide",
        help="Q-value level to filter on (default: peptide)",
    )
    args = parser.parse_args()

    out = write_summary(args.inputs, args.output, args.fdr, args.level)
    if out:
        print(out, file=sys.stderr)


if __name__ == "__main__":
    main()
