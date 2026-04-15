"""Record a SAGE or FragPipe search run into a SQLite regression database.

The caller (Snakemake rule) passes a run_info.json sidecar written by the
respective search rule, plus the FDR-filtered summary file.  All metadata
(tool call, config content, pipeline call) lives in that JSON so this script
stays engine-agnostic.

Required environment variable: ION_MAIDEN_REGRESSION_DB — path to the SQLite DB.
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ---------- schema ---------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_path    TEXT NOT NULL,
    run_date         TEXT NOT NULL,
    search_engine    TEXT NOT NULL,
    dataset          TEXT NOT NULL,
    cfg              TEXT NOT NULL,
    pipeline_config  TEXT,
    search_config    TEXT,
    pipeline_call    TEXT,
    search_tool_call TEXT,
    psm_count        INTEGER,
    peptide_count    INTEGER,
    ion_count        INTEGER,
    protein_count    INTEGER
);
CREATE TABLE IF NOT EXISTS git_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    repo_name    TEXT NOT NULL,
    repo_url     TEXT NOT NULL,
    commit_hash  TEXT NOT NULL
);
"""

# ---------- summary parsers ------------------------------------------------


def _parse_sage_summary(path: Path) -> dict[str, int]:
    lines = path.read_text().splitlines()
    keys = lines[0].split("\t")
    vals = lines[1].split("\t")
    return {k: int(v) for k, v in zip(keys, vals)}


def _parse_fragpipe_summary(path: Path) -> dict[str, int]:
    text = path.read_text()
    line = next(
        (ln for ln in text.splitlines() if "Final report numbers after FDR" in ln),
        None,
    )
    if line is None:
        raise ValueError(f"Could not find FDR summary line in {path}")
    metrics = {}
    for m in re.finditer(r"\b(ions|peptides|proteins|psms)=(\d+)", line):
        metrics[m.group(1)] = int(m.group(2))
    if not metrics:
        raise ValueError(f"No metrics parsed from FDR summary line: {line!r}")
    # normalise key names to match sage columns
    return {
        "psm_count": metrics.get("psms"),
        "peptide_count": metrics.get("peptides"),
        "ion_count": metrics.get("ions"),
        "protein_count": metrics.get("proteins"),
    }


# ---------- git snapshot ---------------------------------------------------


def _capture_git_snapshot(git_root: Path) -> list[dict]:
    pipeline_path = git_root.parent
    get_commits = pipeline_path / "git" / "multigit" / "get_commits"
    if not get_commits.exists():
        raise FileNotFoundError(f"get_commits not found: {get_commits}")
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [str(get_commits), tmp_path, "--git-root", str(git_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in Path(tmp_path).read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            name, url, typ, value = parts
            if typ != "commit":
                continue
            rows.append({"repo_name": name, "repo_url": url, "commit_hash": value})
        return rows
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------- DB write -------------------------------------------------------


def _write_to_db(
    db_path: Path,
    *,
    pipeline_path: str,
    search_engine: str,
    dataset: str,
    cfg: str,
    pipeline_config: str | None,
    search_config: str | None,
    pipeline_call: str | None,
    search_tool_call: str | None,
    counts: dict[str, int | None],
    git_rows: list[dict],
) -> int:
    con = sqlite3.connect(db_path)
    con.executescript(_DDL)
    run_date = datetime.now(timezone.utc).isoformat()
    with con:
        cur = con.execute(
            """
            INSERT INTO runs
                (pipeline_path, run_date, search_engine, dataset, cfg,
                 pipeline_config, search_config, pipeline_call, search_tool_call,
                 psm_count, peptide_count, ion_count, protein_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pipeline_path,
                run_date,
                search_engine,
                dataset,
                cfg,
                pipeline_config,
                search_config,
                pipeline_call,
                search_tool_call,
                counts.get("psm_count"),
                counts.get("peptide_count"),
                counts.get("ion_count"),
                counts.get("protein_count"),
            ),
        )
        run_id = cur.lastrowid
        con.executemany(
            """INSERT INTO git_snapshots
               (run_id, repo_name, repo_url, commit_hash) VALUES (?, ?, ?, ?)""",
            [
                (run_id, r["repo_name"], r["repo_url"], r["commit_hash"])
                for r in git_rows
            ],
        )
    con.close()
    return run_id


# ---------- CLI ------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Record a SAGE or FragPipe run into the regression SQLite DB."
    )
    p.add_argument(
        "--init-db",
        action="store_true",
        help="Create/initialise the DB schema and exit (no run recorded).",
    )
    p.add_argument(
        "--run-info",
        type=Path,
        help="JSON sidecar written by the Snakemake search rule.",
    )
    p.add_argument(
        "--summary",
        type=Path,
        help="FDR-filtered summary file (sage_summary.tsv or fragpipe log_summary.txt).",
    )
    p.add_argument(
        "--git-root",
        type=Path,
        help="Path to the git/ directory holding all sub-repos.",
    )
    p.add_argument(
        "--marker",
        type=Path,
        help="Output marker file; Snakemake uses this to track completion.",
    )
    args = p.parse_args()

    db_path = os.environ.get("ION_MAIDEN_REGRESSION_DB")
    if not db_path:
        raise SystemExit(
            "ION_MAIDEN_REGRESSION_DB environment variable not set. "
            "Export it to the path where the SQLite DB should live."
        )

    if args.init_db:
        con = sqlite3.connect(db_path)
        con.executescript(_DDL)
        con.close()
        print(f"Regression DB initialised at {db_path}")
        return

    if not args.run_info or not args.summary or not args.git_root or not args.marker:
        p.error("--run-info, --summary, --git-root, and --marker are required unless --init-db is used.")

    run_info = json.loads(args.run_info.read_text())
    search_engine = run_info["search_engine"]

    pipeline_config = None
    cfg_path = run_info.get("pipeline_config_path")
    if cfg_path:
        try:
            pipeline_config = Path(cfg_path).read_text()
        except OSError:
            pass

    search_config = None
    sc_path = run_info.get("search_config_path")
    if sc_path:
        try:
            search_config = Path(sc_path).read_text()
        except OSError:
            pass

    if search_engine == "sage":
        counts = _parse_sage_summary(args.summary)
    else:
        counts = _parse_fragpipe_summary(args.summary)

    git_rows = _capture_git_snapshot(args.git_root)

    run_id = _write_to_db(
        Path(db_path),
        pipeline_path=str(Path(".").resolve()),
        search_engine=search_engine,
        dataset=run_info["dataset"],
        cfg=run_info["cfg"],
        pipeline_config=pipeline_config,
        search_config=search_config,
        pipeline_call=run_info.get("pipeline_call"),
        search_tool_call=run_info.get("search_tool_call"),
        counts=counts,
        git_rows=git_rows,
    )

    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(f"run_id={run_id}\n")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
