"""Filter+summarize raw (unfiltered) SAGE results into a single-row count TSV.

Unlike sage_summarize.py (which counts a table someone already FDR-filtered via
sage_filter), this reads SAGE's own raw results table directly -- any format
pandas_ops.io.read_df supports -- and applies the FDR filter itself, in one pass.
"""

import argparse
from pathlib import Path

from searchops.sage import count_sage_at_fdr


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw SAGE results at an FDR threshold and count PSMs/peptides/ions/proteins."
    )
    parser.add_argument("input", type=Path, help="Input raw SAGE results table (e.g. results.sage.tsv)")
    parser.add_argument("output", type=Path, help="Output summary .tsv file")
    parser.add_argument("--fdr", type=float, default=0.01, help="Peptide-level FDR threshold (default: 0.01)")
    parser.add_argument(
        "--level", choices=["psm", "peptide", "protein"], default="peptide",
        help="Q-value level to filter on (default: peptide)",
    )
    args = parser.parse_args()

    counts = count_sage_at_fdr(args.input, fdr=args.fdr, level=args.level)
    with open(args.output, "w") as fh:
        fh.write("\t".join(counts.keys()) + "\n")
        fh.write("\t".join(str(v) for v in counts.values()) + "\n")


if __name__ == "__main__":
    main()
