"""SQLite record of search runs: one row per SAGE or mokapot pipeline node that ran.

The database path comes from the `ION_MAIDEN_REGRESSION_DB` environment variable. A row
holds when the node ran, its dataset, the tool command, the config of every node
upstream of it, the git state of the pipeline checkout, its confident counts, and a
comment filled in afterwards. Schema and usage: git/searchops `docs/ai/run_db.md`.
"""

import json
import os
import sqlite3
import subprocess
import tomllib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ENV_VAR = "ION_MAIDEN_REGRESSION_DB"

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT NOT NULL,
    tool          TEXT NOT NULL,
    node          TEXT NOT NULL,
    pipeline_root TEXT NOT NULL,
    dataset       TEXT,
    command       TEXT NOT NULL,
    config        TEXT NOT NULL,
    git           TEXT NOT NULL,
    fdr           REAL NOT NULL,
    spectra       INTEGER,
    peptidoforms  INTEGER,
    ions          INTEGER,
    comment       TEXT,
    commented_at  TEXT
);
"""

_LEVELS = ("spectra", "peptidoforms", "ions")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the schema if needed. WAL mode and a 60 s busy
    timeout let several pipeline processes write to it at once."""
    con = sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(_DDL)
    return con


def _is_scalar_type(type_name: str) -> bool:
    return all(part.startswith("builtins.") for part in type_name.split("|"))


def ancestor_nodes(parent_paths: list[str | Path]) -> dict[str, dict]:
    """Rule, own config and realized command of every node upstream of `parent_paths`.

    `parent_paths` are output paths inside a necroflow node store
    (`<nodes>/<rule>/<hash>/<file>`). Each node's `.rip/dependencies.toml` holds its
    config merged over all its ancestors' configs; its own config is the part named by
    its rule's scalar parameters. Keyed by `<rule>/<hash>`.
    """
    nodes: dict[str, dict] = {}
    pending = [Path(p) for p in parent_paths]
    while pending:
        node_dir = pending.pop().parent
        nodes_dir = node_dir.parent.parent
        key = node_dir.relative_to(nodes_dir).as_posix()
        if key in nodes:
            continue
        deps = tomllib.loads((node_dir / ".rip" / "dependencies.toml").read_text())
        merged = deps.get("config", {})
        input_types = deps.get("recipe", {}).get("input_types", {})
        nodes[key] = {
            "rule": deps["rule"],
            "config": {k: merged[k] for k, t in input_types.items() if k in merged and _is_scalar_type(t)},
            "command": deps.get("command", {}).get("realized"),
        }
        pending += [nodes_dir / parent["node_key"] for parent in deps.get("parents", [])]
    return nodes


def dataset_of(nodes: dict[str, dict]) -> str | None:
    """The Bruker `.d` path(s) the nodes were computed from (`source_bruker_d`'s `path`)."""
    paths = sorted({n["config"]["path"] for n in nodes.values() if n["rule"] == "source_bruker_d"})
    return ", ".join(paths) or None


def git_snapshot(root: str | Path) -> list[dict]:
    """Branch, commit and dirtiness (tracked files only) of `root` and each repo in `root/git`."""
    root = Path(root)
    git_dir = root / "git"
    repos = [root] + sorted(p for p in git_dir.iterdir() if (p / ".git").exists()) if git_dir.is_dir() else [root]
    snapshot = []
    for repo in repos:

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            ).stdout.strip()

        snapshot.append(
            {
                "path": repo.relative_to(root).as_posix(),
                "branch": git("branch", "--show-current"),
                "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            }
        )
    return snapshot


def record_run(
    db_path: str | Path,
    *,
    tool: str,
    node_dir: str | Path,
    parent_paths: list[str | Path],
    command: str,
    fdr: float,
    levels: dict[str, int],
    root: str | Path | None = None,
) -> int:
    """Insert one run and return its id. `root` is the pipeline checkout, by default the
    working directory necroflow runs commands in."""
    root = Path(root) if root is not None else Path.cwd()
    nodes = ancestor_nodes(parent_paths)
    row = {
        "recorded_at": _now(),
        "tool": tool,
        "node": os.path.relpath(node_dir, root),
        "pipeline_root": str(root),
        "dataset": dataset_of(nodes),
        "command": command,
        "config": json.dumps(nodes, sort_keys=True),
        "git": json.dumps(git_snapshot(root)),
        "fdr": fdr,
        **{level: levels.get(level) for level in _LEVELS},
    }
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    with closing(connect(db_path)) as con, con:
        cursor = con.execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders})", tuple(row.values()))
    return cursor.lastrowid


def set_comment(db_path: str | Path, run_id: int, comment: str) -> None:
    with closing(connect(db_path)) as con, con:
        cursor = con.execute(
            "UPDATE runs SET comment = ?, commented_at = ? WHERE id = ?", (comment, _now(), run_id)
        )
    if cursor.rowcount != 1:
        raise ValueError(f"no run with id {run_id} in {db_path}")


def recent_runs(db_path: str | Path, limit: int = 20) -> list[sqlite3.Row]:
    with closing(connect(db_path)) as con:
        con.row_factory = sqlite3.Row
        return con.execute(
            "SELECT id, recorded_at, tool, dataset, node, fdr, spectra, peptidoforms, ions, comment"
            " FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
