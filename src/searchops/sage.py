"""Parse and summarize SAGE proteomics search engine results."""

from pathlib import Path
from typing import Literal

import duckdb

QCOL = {"psm": "spectrum_q", "peptide": "peptide_q", "protein": "protein_q"}


def summarize_sage(
    path: str | Path,
    fdr: float = 0.01,
    level: Literal["psm", "peptide", "protein"] = "peptide",
) -> dict[str, int]:
    """Summarize SAGE results at a given FDR threshold.

    Args:
        path: Path to results.sage.tsv or results.sage.parquet
        fdr: FDR threshold (q-value cutoff)
        level: Which q-value column gates the filter:
               "psm" -> spectrum_q, "peptide" -> peptide_q, "protein" -> protein_q

    Returns:
        Dict with psm_count, peptide_count, ion_count, protein_count
    """
    q = QCOL[level]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    con = duckdb.connect()
    if path.suffix == ".parquet":
        source = f"read_parquet('{path}')"
    else:
        # quote='' disables quoting; ignore_errors skips rows where the scannr
        # field contains embedded newlines that confuse the column count.
        source = f"read_csv('{path}', delim='\t', header=true, quote='', ignore_errors=true)"

    con.execute(f"CREATE VIEW sage AS SELECT * FROM {source} WHERE {q} <= {fdr}")
    psm_count = con.execute("SELECT COUNT(*) FROM sage").fetchone()[0]
    peptide_count = con.execute("SELECT COUNT(DISTINCT peptide) FROM sage").fetchone()[0]
    ion_count = con.execute(
        "SELECT COUNT(DISTINCT (peptide, charge)) FROM sage"
    ).fetchone()[0]
    protein_count = con.execute("""
        SELECT COUNT(DISTINCT trim(p))
        FROM sage, unnest(string_split(proteins, ';')) AS t(p)
    """).fetchone()[0]
    return {
        "psm_count": psm_count,
        "peptide_count": peptide_count,
        "ion_count": ion_count,
        "protein_count": protein_count,
    }


def count_sage(path: str | Path) -> dict[str, int]:
    """Count PSMs, peptides, ions, and proteins in a pre-filtered sage parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW sage AS SELECT * FROM read_parquet('{path}')")
    psm_count = con.execute("SELECT COUNT(*) FROM sage").fetchone()[0]
    peptide_count = con.execute("SELECT COUNT(DISTINCT peptide) FROM sage").fetchone()[0]
    ion_count = con.execute(
        "SELECT COUNT(DISTINCT (peptide, charge)) FROM sage"
    ).fetchone()[0]
    protein_count = con.execute("""
        SELECT COUNT(DISTINCT trim(p))
        FROM sage, unnest(string_split(proteins, ';')) AS t(p)
    """).fetchone()[0]
    return {
        "psm_count": psm_count,
        "peptide_count": peptide_count,
        "ion_count": ion_count,
        "protein_count": protein_count,
    }
