"""Manage the run database at $ION_MAIDEN_REGRESSION_DB.

Runs are recorded by `sage-fdr-levels`/`mokapot-fdr-levels` inside the pipeline's
`run_sage`/`mokapot` commands; this CLI creates the database, lists runs and sets a
run's comment.
"""

import argparse
import os

from searchops import run_db


def main():
    parser = argparse.ArgumentParser(description=f"Manage the run database at ${run_db.ENV_VAR}.")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init", help="Create the database and its schema")
    listing = commands.add_parser("list", help="Show the latest runs")
    listing.add_argument("-n", type=int, default=20, help="How many (default: 20)")
    comment = commands.add_parser("comment", help="Set a run's comment")
    comment.add_argument("run_id", type=int)
    comment.add_argument("text")
    args = parser.parse_args()

    db_path = os.environ.get(run_db.ENV_VAR)
    if not db_path:
        raise SystemExit(f"{run_db.ENV_VAR} is not set: export it as the path of the SQLite database.")

    if args.action == "init":
        run_db.connect(db_path).close()
        print(f"run database ready at {db_path}")
    elif args.action == "list":
        rows = run_db.recent_runs(db_path, args.n)
        if rows:
            print("\t".join(rows[0].keys()))
        for row in rows:
            print("\t".join("" if v is None else str(v) for v in row))
    elif args.action == "comment":
        run_db.set_comment(db_path, args.run_id, args.text)
        print(f"comment set on run {args.run_id}")


if __name__ == "__main__":
    main()
