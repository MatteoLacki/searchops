"""Confident target counts at one FDR, each level from its own target-decoy competition.

Three levels per programme: spectra (one PSM per spectrum), peptidoforms (peptide
strings, modifications included) and ions ((peptide, charge) pairs, ion-level FDR).
Counted in DuckDB SQL straight from the result tables.
"""

from pathlib import Path

import duckdb

from searchops.ion_level_fdr import mokapot_ion_level_ids, read_table, sage_ion_level_ids


def sage_fdr_levels(results_tsv: str | Path, fdr: float = 0.01) -> dict[str, int]:
    """Spectra (`spectrum_q`), peptidoforms (`peptide_q`) and ions in SAGE's results.

    Only rank-1 target PSMs count. Ions are counted at ion-level FDR
    (`ion_level_fdr.sage_ion_level_ids`), since SAGE reports no ion-level q-value.
    """
    spectra, peptidoforms = duckdb.connect().execute(
        f"""
        SELECT count(*) FILTER (WHERE spectrum_q <= $fdr),
               count(DISTINCT peptide) FILTER (WHERE peptide_q <= $fdr)
        FROM {read_table(results_tsv)}
        WHERE rank = 1 AND label = 1
        """,
        {"fdr": fdr},
    ).fetchone()
    _, ions = sage_ion_level_ids(results_tsv, fdr)
    return {"spectra": spectra, "peptidoforms": peptidoforms, "ions": len(ions)}


def mokapot_fdr_levels(
    psms_txt: str | Path,
    peptides_txt: str | Path,
    decoy_psms_txt: str | Path,
    sage_results_tsv: str | Path,
    fdr: float = 0.01,
) -> dict[str, int]:
    """Spectra (`mokapot.psms.txt`), peptidoforms (`mokapot.peptides.txt`) and ions.

    Spectra and peptidoforms use mokapot's own q-values. Ions are counted at ion-level
    FDR from mokapot's target and decoy PSM scores (`ion_level_fdr.mokapot_ion_level_ids`).
    """
    spectra, peptidoforms = duckdb.connect().execute(
        f"""
        SELECT
            (SELECT count(*) FROM {read_table(psms_txt)}
             WHERE Label::BOOLEAN AND "mokapot q-value" <= $fdr),
            (SELECT count(DISTINCT Peptide) FROM {read_table(peptides_txt)}
             WHERE Label::BOOLEAN AND "mokapot q-value" <= $fdr)
        """,
        {"fdr": fdr},
    ).fetchone()
    _, ions = mokapot_ion_level_ids(psms_txt, decoy_psms_txt, sage_results_tsv, fdr)
    return {"spectra": spectra, "peptidoforms": peptidoforms, "ions": len(ions)}
