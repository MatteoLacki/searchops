"""CLI to compare SAGE results across multiple experiment folders."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from searchops.sage import summarize_sage

COLUMNS = ["label", "psm_count", "peptide_count", "ion_count", "protein_count"]


def collect_results(folders, fdr, level, glob_pattern):
    # Resolve all (file, label) pairs first so tqdm knows the total
    work = []
    for arg in folders:
        path = Path(arg)
        if path.is_file():
            work.append((path, str(path)))
        else:
            tsv_files = sorted(path.glob(glob_pattern))
            if not tsv_files:
                print(f"Warning: no files matching '{glob_pattern}' in {path}", file=sys.stderr)
                continue
            for tsv in tsv_files:
                work.append((tsv, str(tsv.relative_to(path))))

    rows = []
    for tsv, label in tqdm(work, unit="file"):
        counts = summarize_sage(tsv, fdr=fdr, level=level)
        rows.append({"label": label, **counts})
    return rows


def print_table(rows, sort_col):
    if not rows:
        print("No results found.")
        return
    print(pd.DataFrame(rows, columns=COLUMNS).sort_values(sort_col, ascending=False).to_csv(index=False))


def print_json(rows):
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and compare SAGE search results across experiment folders."
    )
    parser.add_argument("folders", nargs="+", help="Experiment folder(s) to scan")
    parser.add_argument("--fdr", type=float, default=0.01, help="FDR threshold (default: 0.01)")
    parser.add_argument(
        "--level",
        choices=["psm", "peptide", "protein"],
        default="peptide",
        help="Q-value level to filter on (default: peptide)",
    )
    parser.add_argument(
        "--glob",
        default="**/results.sage.tsv",
        help="Glob pattern to find result files within each folder (default: **/results.sage.tsv)",
    )
    parser.add_argument(
        "--output",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    rows = collect_results(args.folders, args.fdr, args.level, args.glob)

    sort_col = f"{args.level}_count"
    if args.output in ("table", "csv"):
        print_table(rows, sort_col)
    elif args.output == "json":
        print_json(rows)


if __name__ == "__main__":
    main()
