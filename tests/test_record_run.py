import sqlite3

from searchops.cli import record_run


def _insert_run(db_path, *, dataset="F9477", cfg="optimal2tier", peptides=10):
    return record_run._write_to_db(
        db_path,
        pipeline_path="/pipeline",
        search_engine="sage",
        dataset=dataset,
        cfg=cfg,
        pipeline_config=None,
        search_config=None,
        pipeline_call=None,
        search_tool_call=None,
        counts={
            "psm_count": peptides * 2,
            "peptide_count": peptides,
            "ion_count": peptides + 1,
            "protein_count": peptides - 1,
        },
        accepted=False,
        acceptance_reason=None,
        status="unreviewed",
        regression=False,
        message=None,
        git_rows=[],
    )


def _marker(tmp_path, run_id):
    marker = tmp_path / f"run_{run_id}.marker"
    marker.write_text(f"run_id={run_id}\n")
    return marker


def test_old_accepted_rows_migrate_to_reviewed_non_regression(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_path TEXT NOT NULL,
            run_date TEXT NOT NULL,
            search_engine TEXT NOT NULL,
            dataset TEXT NOT NULL,
            cfg TEXT NOT NULL,
            psm_count INTEGER,
            peptide_count INTEGER,
            ion_count INTEGER,
            protein_count INTEGER,
            accepted INTEGER NOT NULL DEFAULT 0,
            acceptance_reason TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO runs
            (pipeline_path, run_date, search_engine, dataset, cfg,
             psm_count, peptide_count, ion_count, protein_count,
             accepted, acceptance_reason)
        VALUES ('/pipeline', 'now', 'sage', 'F9477', 'optimal2tier',
                20, 10, 11, 9, 1, 'known good')
        """
    )
    con.commit()

    record_run._ensure_schema(con)

    row = con.execute(
        "SELECT status, regression, message FROM runs WHERE id = 1"
    ).fetchone()
    con.close()
    assert row == ("reviewed", 0, "known good")


def test_new_record_defaults_to_unreviewed_non_regression(tmp_path):
    db_path = tmp_path / "runs.sqlite"

    run_id = _insert_run(db_path)

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT status, regression, message FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    con.close()
    assert row == ("unreviewed", 0, None)


def test_review_marker_updates_status_regression_and_message(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    run_id = _insert_run(db_path)

    updated_id = record_run._review_run(
        db_path,
        marker=_marker(tmp_path, run_id),
        regression=True,
        message="peptide count dropped",
    )

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT status, regression, message FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    con.close()
    assert updated_id == run_id
    assert row == ("reviewed", 1, "peptide count dropped")


def test_mark_bested_marks_only_matching_unreviewed_previous_runs(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    stale_id = _insert_run(db_path, peptides=10)
    reviewed_id = _insert_run(db_path, peptides=11)
    other_cfg_id = _insert_run(db_path, cfg="other", peptides=12)
    current_id = _insert_run(db_path, peptides=20)
    record_run._review_run(
        db_path,
        marker=_marker(tmp_path, reviewed_id),
        regression=False,
        message="keep reviewed",
    )

    count = record_run._mark_bested_for_marker(
        db_path,
        marker=_marker(tmp_path, current_id),
        message="bested by current",
    )

    con = sqlite3.connect(db_path)
    rows = dict(con.execute("SELECT id, status FROM runs").fetchall())
    messages = dict(con.execute("SELECT id, message FROM runs").fetchall())
    con.close()
    assert count == 1
    assert rows[stale_id] == "bested"
    assert messages[stale_id] == "bested by current"
    assert rows[reviewed_id] == "reviewed"
    assert rows[other_cfg_id] == "unreviewed"
    assert rows[current_id] == "unreviewed"
