import pandas as pd

from searchops.fdr_levels import mokapot_fdr_levels, sage_fdr_levels


def _sage_tsv(tmp_path, rows):
    path = tmp_path / "results.sage.tsv"
    pd.DataFrame(
        rows,
        columns=["psm_id", "peptide", "charge", "label", "rank", "sage_discriminant_score", "spectrum_q", "peptide_q"],
    ).to_csv(path, sep="\t", index=False)
    return path


# Eight target ions scored 10..3 and two decoy ions below them: at fdr = 0.2 every target
# ion passes. PEP0 has a second, weaker spectrum, and a charge-3 ion scored below the decoys.
# A rank-2 PSM and a decoy never count as spectra or peptidoforms.
def _rows():
    rows = [(i, f"PEP{i}", 2, 1, 1, 10.0 - i, 0.001, 0.001) for i in range(8)]
    rows += [
        (8, "PEP0", 2, 1, 1, 0.5, 0.15, 0.001),
        (9, "PEP0", 3, 1, 1, 0.1, 0.3, 0.001),
        (10, "DECOYA", 2, -1, 1, 2.5, 1.0, 1.0),
        (11, "DECOYB", 2, -1, 1, 1.5, 1.0, 1.0),
        (12, "RANKTWO", 2, 1, 2, 100.0, 0.001, 0.001),
    ]
    return rows


def test_sage_fdr_levels(tmp_path):
    levels = sage_fdr_levels(_sage_tsv(tmp_path, _rows()), fdr=0.2)
    assert levels == {"spectra": 9, "peptidoforms": 8, "ions": 8}


def _mokapot_txt(path, rows):
    pd.DataFrame(
        rows, columns=["SpecId", "Label", "ScanNr", "Peptide", "mokapot score", "mokapot q-value", "mokapot PEP", "Proteins"]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_mokapot_fdr_levels(tmp_path):
    rows = _rows()
    sage = _sage_tsv(tmp_path, rows)
    rank_one = [r for r in rows if r[4] == 1]
    # mokapot's own q-values: SAGE's spectrum_q for PSMs; peptides get the best PSM's.
    targets = [(r[0], "True", "", r[1], r[5], r[6], 0.0, "P") for r in rank_one if r[3] == 1]
    decoys = [(r[0], "False", "", r[1], r[5], r[6], 0.0, "rev_P") for r in rank_one if r[3] == -1]
    psms = _mokapot_txt(tmp_path / "mokapot.psms.txt", targets)
    decoy_psms = _mokapot_txt(tmp_path / "mokapot.decoy.psms.txt", decoys)
    peptides = _mokapot_txt(tmp_path / "mokapot.peptides.txt", [t for t in targets if t[0] < 8])
    levels = mokapot_fdr_levels(psms, peptides, decoy_psms, sage, fdr=0.2)
    assert levels == {"spectra": 9, "peptidoforms": 8, "ions": 8}
