"""Ion-level FDR, the way DIA-NN controls it for precursors, in DuckDB SQL.

An ion is a (peptide, charge) pair. Each ion is scored by its best PSM, target and decoy
ions are ranked together by that score, and target-decoy competition gives each ion a
q-value. The confident peptides are the distinct peptides of the ions that pass.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def read_table(path: str | Path) -> str:
    """SQL reading a results table: `read_parquet` for `.parquet`, else tab-separated `read_csv`."""
    literal = "'" + str(path).replace("'", "''") + "'"
    if Path(path).suffix == ".parquet":
        return f"read_parquet({literal})"
    return f"read_csv({literal}, delim='\\t', header=true)"


def qvalues_query(scored: str) -> str:
    """SQL adding target-decoy competition q-values `q` to `scored`.

    `scored` is a query with a `score` column (higher is better) and a boolean
    `is_target`. The FDR at a score threshold is (decoys + 1) / targets among the rows
    scoring at least that much; the RANGE frames give tied scores the counts of their
    whole tie block. A row's q-value is the lowest FDR at its score or any lower one,
    capped at 1. This is the estimate of mokapot.qvalues.tdc.
    """
    return f"""
        WITH counted AS (
            SELECT *,
                sum(is_target::INTEGER) OVER at_least AS targets,
                sum((NOT is_target)::INTEGER) OVER at_least AS decoys
            FROM ({scored})
            WINDOW at_least AS (ORDER BY score DESC RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT * EXCLUDE (targets, decoys),
            min(CASE WHEN targets > 0 THEN least((decoys + 1) / targets, 1.0) ELSE 1.0 END)
                OVER (ORDER BY score RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS q
        FROM counted
    """


def tdc_qvalues(scores: np.ndarray, is_target: np.ndarray) -> np.ndarray:
    """`qvalues_query` on arrays: one q-value per entry, in input order."""
    entries = pd.DataFrame(
        {"entry": np.arange(len(scores)), "score": np.asarray(scores, dtype=np.float64),
         "is_target": np.asarray(is_target, dtype=bool)}
    )
    con = duckdb.connect()
    con.register("entries", entries)
    return con.execute(f"SELECT q FROM ({qvalues_query('SELECT * FROM entries')}) ORDER BY entry").fetchnumpy()["q"]


def _peptides_and_ions(rows: list[tuple[str, int]]) -> tuple[set[str], set[tuple[str, int]]]:
    return {peptide for peptide, _ in rows}, set(rows)


def sage_ion_level_ids(results_tsv: str | Path, fdr: float = 0.01) -> tuple[set[str], set[tuple[str, int]]]:
    """Confident target peptides and ions in SAGE's results at ion-level FDR.

    Only rank-1 PSMs count. Each ion is scored by its best `sage_discriminant_score`;
    target (`label = 1`) and decoy (`label = -1`) ions are ranked together.
    """
    scored = f"""
        SELECT peptide, charge, label = 1 AS is_target, max(sage_discriminant_score) AS score
        FROM {read_table(results_tsv)}
        WHERE rank = 1
        GROUP BY peptide, charge, label
    """
    rows = duckdb.connect().execute(
        f"SELECT peptide, charge FROM ({qvalues_query(scored)}) WHERE is_target AND q <= ?", [fdr]
    ).fetchall()
    return _peptides_and_ions(rows)


def mokapot_ion_level_ids(
    psms_txt: str | Path,
    decoy_psms_txt: str | Path,
    sage_results_tsv: str | Path,
    fdr: float = 0.01,
) -> tuple[set[str], set[tuple[str, int]]]:
    """Confident target peptides and ions in mokapot's results at ion-level FDR.

    `psms_txt` and `decoy_psms_txt` are mokapot's target and decoy PSM tables
    (`mokapot.psms.txt`, and `mokapot.decoy.psms.txt`, which mokapot writes only with
    `--keep_decoys`). Each ion is scored by its best `mokapot score`. mokapot does not
    report charge, so each PSM takes it from the SAGE results it was rescored from,
    joined on `SpecId` = `psm_id`.
    """
    con = duckdb.connect()
    con.execute(f"""
        CREATE TEMP TABLE psms AS
        SELECT p.Peptide AS peptide, p.is_target, p.score, s.peptide AS sage_peptide, s.charge
        FROM (
            SELECT SpecId, Peptide, Label::BOOLEAN AS is_target, "mokapot score" AS score FROM {read_table(psms_txt)}
            UNION ALL
            SELECT SpecId, Peptide, Label::BOOLEAN AS is_target, "mokapot score" AS score FROM {read_table(decoy_psms_txt)}
        ) AS p
        LEFT JOIN {read_table(sage_results_tsv)} AS s ON p.SpecId = s.psm_id
    """)
    missing, differ = con.execute(
        "SELECT count(*) FILTER (WHERE sage_peptide IS NULL), count(*) FILTER (WHERE sage_peptide <> peptide) FROM psms"
    ).fetchone()
    if missing:
        raise ValueError(f"{missing} mokapot SpecIds are not psm_ids in {sage_results_tsv}")
    if differ:
        raise ValueError(f"mokapot SpecId does not join to {sage_results_tsv}'s psm_id: peptides differ")
    scored = "SELECT peptide, charge, is_target, max(score) AS score FROM psms GROUP BY peptide, charge, is_target"
    rows = con.execute(
        f"SELECT peptide, charge FROM ({qvalues_query(scored)}) WHERE is_target AND q <= ?", [fdr]
    ).fetchall()
    return _peptides_and_ions(rows)
