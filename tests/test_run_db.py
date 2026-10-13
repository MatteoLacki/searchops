import json
import sqlite3
import subprocess

import pytest

from searchops import run_db


def _node(nodes, key, filename, rule, config, input_types, parents=(), command="cmd"):
    node_dir = nodes / key
    (node_dir / ".rip").mkdir(parents=True)
    (node_dir / filename).write_text("")
    lines = [f'rule = "{rule}"', "", "[config]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in config.items()]
    for parent in parents:
        lines += ["", "[[parents]]", f'node_key = "{parent}"']
    lines += ["", "[recipe.input_types]"]
    lines += [f'{k} = "{v}"' for k, v in input_types.items()]
    lines += ["", "[command]", f'realized = "{command}"']
    (node_dir / ".rip" / "dependencies.toml").write_text("\n".join(lines) + "\n")
    return node_dir / filename


@pytest.fixture
def node_store(tmp_path):
    nodes = tmp_path / "nodes"
    _node(nodes, "source_bruker_d/a", "input.d", "source_bruker_d", {"path": "data/X.d"}, {"path": "builtins.str"})
    # Merged config: the ancestor's `path` plus this rule's own `fdr`; only `fdr` is its own.
    precursors = _node(
        nodes, "pick/b", "precursors.mmappet", "pick", {"path": "data/X.d", "fdr": 0.01},
        {"tdf": "pkg.BrukerD", "fdr": "builtins.float"}, parents=["source_bruker_d/a/input.d"],
    )
    return tmp_path, precursors


def test_ancestor_nodes_keeps_each_nodes_own_config(node_store):
    _, precursors = node_store
    nodes = run_db.ancestor_nodes([precursors])
    assert nodes == {
        "pick/b": {"rule": "pick", "config": {"fdr": 0.01}, "command": "cmd"},
        "source_bruker_d/a": {"rule": "source_bruker_d", "config": {"path": "data/X.d"}, "command": "cmd"},
    }
    assert run_db.dataset_of(nodes) == "data/X.d"


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)


def test_git_snapshot(tmp_path):
    sub = tmp_path / "git" / "sub"
    sub.mkdir(parents=True)
    for repo in (tmp_path, sub):
        _git(repo, "init", "-b", "main")
        (repo / "f").write_text("x")
        _git(repo, "add", "f")
        _git(repo, "commit", "-m", "c")
    (sub / "f").write_text("changed")
    snapshot = run_db.git_snapshot(tmp_path)
    assert [(s["path"], s["branch"], s["dirty"]) for s in snapshot] == [(".", "main", False), ("git/sub", "main", True)]
    assert all(len(s["commit"]) == 40 for s in snapshot)


def test_record_run_and_comment(node_store, monkeypatch):
    root, precursors = node_store
    monkeypatch.setattr(run_db, "git_snapshot", lambda root: [{"path": ".", "branch": "main", "commit": "c" * 40, "dirty": False}])
    db = root / "runs.sqlite"
    run_id = run_db.record_run(
        db, tool="sage", node_dir=root / "nodes" / "run_sage" / "c", parent_paths=[precursors],
        command="sage -f x", fdr=0.01, levels={"spectra": 3, "peptidoforms": 2, "ions": 2}, root=root,
    )
    run_db.set_comment(db, run_id, "looks fine")
    row = sqlite3.connect(db).execute(
        "SELECT tool, node, dataset, command, spectra, peptidoforms, ions, comment, config FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row[:8] == ("sage", "nodes/run_sage/c", "data/X.d", "sage -f x", 3, 2, 2, "looks fine")
    assert set(json.loads(row[8])) == {"pick/b", "source_bruker_d/a"}
    with pytest.raises(ValueError, match="no run with id"):
        run_db.set_comment(db, run_id + 1, "x")
