"""Count confident spectra, peptidoforms and ions of one SAGE or mokapot run.

Writes a one-row TSV (`fdr`, `spectra`, `peptidoforms`, `ions`). With `--node-dir`
and `ION_MAIDEN_REGRESSION_DB` set, also records the run in that database; a failed
record is reported on stderr and never fails the command.
"""

import argparse
import os
import sys
from pathlib import Path

from searchops import run_db
from searchops.fdr_levels import mokapot_fdr_levels, sage_fdr_levels


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output", type=Path, help="Output one-row TSV")
    parser.add_argument("--fdr", type=float, default=0.01, help="FDR threshold at every level (default: 0.01)")
    parser.add_argument("--node-dir", type=Path, help="Pipeline node directory of the run, for the run database")
    parser.add_argument(
        "--parent", type=Path, action="append", default=[],
        help="Input path of the node (repeatable); its ancestors' configs are recorded",
    )
    parser.add_argument("--command", default="", help="Tool command of the run, as recorded")


def _write_and_record(args: argparse.Namespace, tool: str, levels: dict[str, int]) -> None:
    with open(args.output, "w") as fh:
        fh.write("\t".join(["fdr", *levels]) + "\n")
        fh.write("\t".join(str(v) for v in [args.fdr, *levels.values()]) + "\n")
    db_path = os.environ.get(run_db.ENV_VAR)
    if not db_path or args.node_dir is None:
        return
    try:
        run_id = run_db.record_run(
            db_path, tool=tool, node_dir=args.node_dir, parent_paths=args.parent,
            command=args.command, fdr=args.fdr, levels=levels,
        )
        print(f"recorded run {run_id} in {db_path}")
    except Exception as exc:
        print(f"not recorded in {db_path}: {exc!r}", file=sys.stderr)


def sage_main():
    parser = argparse.ArgumentParser(description="Confident spectra, peptidoforms and ions in SAGE results.")
    parser.add_argument("results_tsv", type=Path, help="SAGE's results.sage.tsv")
    _add_common_arguments(parser)
    args = parser.parse_args()
    _write_and_record(args, "sage", sage_fdr_levels(args.results_tsv, args.fdr))


def mokapot_main():
    parser = argparse.ArgumentParser(description="Confident spectra, peptidoforms and ions in mokapot results.")
    parser.add_argument("psms_txt", type=Path, help="mokapot.psms.txt")
    parser.add_argument("peptides_txt", type=Path, help="mokapot.peptides.txt")
    parser.add_argument("decoy_psms_txt", type=Path, help="mokapot.decoy.psms.txt (mokapot --keep_decoys)")
    parser.add_argument("sage_results_tsv", type=Path, help="SAGE results the PSMs were rescored from (charges)")
    _add_common_arguments(parser)
    args = parser.parse_args()
    levels = mokapot_fdr_levels(
        args.psms_txt, args.peptides_txt, args.decoy_psms_txt, args.sage_results_tsv, args.fdr
    )
    _write_and_record(args, "mokapot", levels)
