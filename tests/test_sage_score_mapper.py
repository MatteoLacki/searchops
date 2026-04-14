import numpy as np

from searchops.cli import sage_score_mapper as mod


def test_build_valid_edges_ignores_nan_values():
    edges, finite_count = mod._build_valid_edges(np.array([1.0, np.nan, 3.0]))

    assert finite_count == 2
    assert np.isfinite(edges).all()
    assert edges[0] == 1.0
    assert edges[-1] == 3.0


def test_make_plots_uses_finite_edges_for_nan_inputs(tmp_path):
    tasks = mod._make_plots(
        tmp_path / "all",
        pmsms_score=np.array([1.0, np.nan, 1.0, 1.0]),
        pmsms_intensity=np.array([99, 999, 9_999, 99_999], dtype=np.uint32),
        matched_scores=np.array([1.0, np.nan, 1.0]),
        matched_intensity=np.array([99, 9_999, 99_999], dtype=np.uint32),
        charge_per_fragment=np.array([2, 3, 4], dtype=object),
        score_label="score",
    )

    dist_kwargs = tasks[0][1]
    assert np.isfinite(dist_kwargs["edges"]).all()
    assert np.diff(dist_kwargs["edges"]).min() > 0
    assert dist_kwargs["all_counts"].sum() == 3
    assert dist_kwargs["matched_counts"].sum() == 2


def test_make_plots_skips_empty_all_slice(tmp_path, capsys):
    tasks = mod._make_plots(
        tmp_path / "empty",
        pmsms_score=np.array([]),
        pmsms_intensity=np.array([], dtype=np.uint32),
        matched_scores=np.array([]),
        matched_intensity=np.array([], dtype=np.uint32),
        charge_per_fragment=np.array([]),
        score_label="score",
    )

    assert tasks == []
    assert "skipping 2D plots" in capsys.readouterr().out


def test_finite_histogram_numba_ignores_nan_values():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    counts, finite_count = mod._finite_histogram_numba(
        np.array([0.2, np.nan, 1.5, 2.8, np.nan]), edges
    )

    assert finite_count == 3
    assert np.array_equal(counts, np.array([1, 1, 1]))


def test_q30_threshold_works_after_filtering():
    matched_scores = np.array([10.0, np.nan, 12.0])
    log10_int_matched = np.array([1.0, np.nan, 9.0])

    finite_q30_matched = np.isfinite(matched_scores) & np.isfinite(log10_int_matched)
    threshold = float(np.quantile(log10_int_matched[finite_q30_matched], 0.30))
    assert np.isfinite(threshold)
    assert threshold >= 1.0
