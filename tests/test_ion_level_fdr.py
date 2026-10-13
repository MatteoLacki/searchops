import numpy as np
import pandas as pd
import pytest

from searchops.ion_level_fdr import mokapot_ion_level_ids, sage_ion_level_ids, tdc_qvalues
from searchops.sage import count_sage_at_fdr


def test_tdc_qvalues_by_hand():
    # Sorted: T T D T D -> FDR (decoys + 1) / targets = 1, 1/2, 1, 2/3, 1.
    q = tdc_qvalues(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), np.array([True, True, False, True, False]))
    assert np.allclose(q, [0.5, 0.5, 2 / 3, 2 / 3, 1.0])


def test_tdc_qvalues_returns_input_order():
    q = tdc_qvalues(np.array([1.0, 3.0, 2.0, 5.0, 4.0]), np.array([False, False, True, True, True]))
    assert np.allclose(q, [1.0, 2 / 3, 2 / 3, 0.5, 0.5])


def test_tdc_qvalues_ties_share_the_fdr_of_the_whole_block():
    # Sorted: T (T D tied) T -> FDR 1, [1/2, 1], 2/3; the tied pair takes 1, not 1/2.
    q = tdc_qvalues(np.array([5.0, 4.0, 4.0, 3.0]), np.array([True, True, False, True]))
    assert np.allclose(q, [2 / 3] * 4)


def test_tdc_qvalues_capped_at_one():
    q = tdc_qvalues(np.array([2.0, 1.0]), np.array([False, False]))
    assert np.allclose(q, [1.0, 1.0])


def _sage_tsv(tmp_path, rows):
    path = tmp_path / "results.sage.tsv"
    pd.DataFrame(
        rows, columns=["psm_id", "peptide", "charge", "label", "rank", "sage_discriminant_score", "peptide_q", "proteins"]
    ).to_csv(path, sep="\t", index=False)
    return path


# Eight target ions with distinct best scores 10..3, then two decoy ions at 2.5 and 1.5:
# (decoys + 1) / targets is 1/8 for the targets, so at fdr = 0.2 every target ion passes and
# at fdr = 0.1 none does. A target PSM at rank 2 with the top score must not count.
def _ion_level_rows():
    rows = [(i, f"PEP{i}", 2, 1, 1, 10.0 - i, 0.001, "P") for i in range(8)]
    rows += [
        (8, "PEP0", 2, 1, 1, 0.5, 0.001, "P"),  # weaker PSM of an ion already scored 10
        (9, "PEP0", 3, 1, 1, 0.1, 0.001, "P"),  # same peptide, another charge, scored below both decoys
        (10, "DECOYA", 2, -1, 1, 2.5, 1.0, "rev_P"),
        (11, "DECOYB", 2, -1, 1, 1.5, 1.0, "rev_P"),
        (12, "RANKTWO", 2, 1, 2, 100.0, 0.001, "P"),
    ]
    return rows


def test_sage_ion_level_ids(tmp_path):
    path = _sage_tsv(tmp_path, _ion_level_rows())
    peptides, ions = sage_ion_level_ids(path, fdr=0.2)
    assert ions == {(f"PEP{i}", 2) for i in range(8)}
    assert peptides == {f"PEP{i}" for i in range(8)}
    assert sage_ion_level_ids(path, fdr=0.1) == (set(), set())


def test_count_sage_at_fdr_counts_rank_one_only(tmp_path):
    path = _sage_tsv(tmp_path, _ion_level_rows())
    counts = count_sage_at_fdr(path, fdr=0.01)
    assert counts["peptide_count"] == 8
    assert counts["ion_count"] == 9


def _mokapot_txt(path, rows):
    pd.DataFrame(
        rows, columns=["SpecId", "Label", "ScanNr", "Peptide", "mokapot score", "mokapot q-value", "mokapot PEP", "Proteins"]
    ).to_csv(path, sep="\t", index=False)
    return path


def test_mokapot_ion_level_ids_takes_charge_from_sage(tmp_path):
    sage_rows = _ion_level_rows()
    sage = _sage_tsv(tmp_path, sage_rows)
    # mokapot rescored every rank-1 PSM: same SpecIds, its own scores (here SAGE's, times 10).
    by_label = {True: [], False: []}
    for psm_id, peptide, _, label, rank, score, _, proteins in sage_rows:
        if rank == 1:
            by_label[label == 1].append((psm_id, str(label == 1), f"precursor_idx={psm_id}", peptide, 10 * score, 0.0, 0.0, proteins))
    targets = _mokapot_txt(tmp_path / "mokapot.psms.txt", by_label[True])
    decoys = _mokapot_txt(tmp_path / "mokapot.decoy.psms.txt", by_label[False])
    peptides, ions = mokapot_ion_level_ids(targets, decoys, sage, fdr=0.2)
    assert ions == {(f"PEP{i}", 2) for i in range(8)}
    assert peptides == {f"PEP{i}" for i in range(8)}


def test_mokapot_ion_level_ids_rejects_a_bad_join(tmp_path):
    sage = _sage_tsv(tmp_path, _ion_level_rows())
    targets = _mokapot_txt(tmp_path / "mokapot.psms.txt", [(0, "True", "", "WRONG", 1.0, 0.0, 0.0, "P")])
    decoys = _mokapot_txt(tmp_path / "mokapot.decoy.psms.txt", [(10, "False", "", "DECOYA", 0.5, 0.0, 0.0, "rev_P")])
    with pytest.raises(ValueError, match="peptides differ"):
        mokapot_ion_level_ids(targets, decoys, sage)
