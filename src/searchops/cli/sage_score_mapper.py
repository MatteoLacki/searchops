import argparse
import textwrap
import tomllib
from pathlib import Path

import plotnine as P
import mmappet
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kilograms as kg

from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

CHUNK_SIZE = 5_000_000
BINS_2D = (200, 200)


def _describe_drop(label: str, before: int, after: int):
    dropped = before - after
    if dropped:
        print(f"{label}: dropped {dropped:_} non-finite rows; kept {after:_}.")


def _expand_if_degenerate(min_value: float, max_value: float):
    if min_value == max_value:
        pad = max(abs(min_value) * 1e-6, 1e-6)
        return min_value - pad, max_value + pad
    return min_value, max_value


@numba.njit(parallel=True, boundscheck=True)
def _finite_min_max_numba(values):
    n_threads = numba.get_num_threads()
    mins = np.full(n_threads, np.inf, dtype=np.float64)
    maxs = np.full(n_threads, -np.inf, dtype=np.float64)
    counts = np.zeros(n_threads, dtype=np.int64)

    for i in numba.prange(len(values)):
        tid = numba.get_thread_id()
        value = values[i]
        if np.isfinite(value):
            if value < mins[tid]:
                mins[tid] = value
            if value > maxs[tid]:
                maxs[tid] = value
            counts[tid] += 1

    found = False
    min_value = 0.0
    max_value = 0.0
    finite_count = 0
    for tid in range(n_threads):
        if counts[tid] == 0:
            continue
        finite_count += counts[tid]
        if not found:
            min_value = mins[tid]
            max_value = maxs[tid]
            found = True
        else:
            if mins[tid] < min_value:
                min_value = mins[tid]
            if maxs[tid] > max_value:
                max_value = maxs[tid]

    return found, min_value, max_value, finite_count


@numba.njit(parallel=True, boundscheck=True)
def _finite_histogram_numba(values, edges):
    n_threads = numba.get_num_threads()
    counts = np.zeros((n_threads, len(edges) - 1), dtype=np.int64)
    finite_counts = np.zeros(n_threads, dtype=np.int64)
    left = edges[0]
    right = edges[-1]
    width = counts.shape[1]

    for i in numba.prange(len(values)):
        tid = numba.get_thread_id()
        value = values[i]
        if not np.isfinite(value):
            continue
        finite_counts[tid] += 1
        if value < left or value > right:
            continue
        if value == right:
            counts[tid, width - 1] += 1
            continue
        idx = np.searchsorted(edges, value, side="right") - 1
        if 0 <= idx < width:
            counts[tid, idx] += 1

    merged = np.zeros(width, dtype=np.int64)
    finite_count = 0
    for tid in range(n_threads):
        merged += counts[tid]
        finite_count += finite_counts[tid]

    return merged, finite_count


@numba.njit(parallel=True, boundscheck=True)
def _finite_min_max_threshold_numba(scores, intensities, threshold):
    n_threads = numba.get_num_threads()
    mins = np.full(n_threads, np.inf, dtype=np.float64)
    maxs = np.full(n_threads, -np.inf, dtype=np.float64)
    counts = np.zeros(n_threads, dtype=np.int64)

    for i in numba.prange(len(scores)):
        tid = numba.get_thread_id()
        value = scores[i]
        if not np.isfinite(value):
            continue
        if np.log10(1.0 + intensities[i]) < threshold:
            continue
        if value < mins[tid]:
            mins[tid] = value
        if value > maxs[tid]:
            maxs[tid] = value
        counts[tid] += 1

    found = False
    min_value = 0.0
    max_value = 0.0
    finite_count = 0
    for tid in range(n_threads):
        if counts[tid] == 0:
            continue
        finite_count += counts[tid]
        if not found:
            min_value = mins[tid]
            max_value = maxs[tid]
            found = True
        else:
            if mins[tid] < min_value:
                min_value = mins[tid]
            if maxs[tid] > max_value:
                max_value = maxs[tid]

    return found, min_value, max_value, finite_count


@numba.njit(parallel=True, boundscheck=True)
def _finite_histogram_threshold_numba(scores, intensities, edges, threshold):
    n_threads = numba.get_num_threads()
    counts = np.zeros((n_threads, len(edges) - 1), dtype=np.int64)
    finite_counts = np.zeros(n_threads, dtype=np.int64)
    left = edges[0]
    right = edges[-1]
    width = counts.shape[1]

    for i in numba.prange(len(scores)):
        tid = numba.get_thread_id()
        value = scores[i]
        if not np.isfinite(value):
            continue
        if np.log10(1.0 + intensities[i]) < threshold:
            continue
        finite_counts[tid] += 1
        if value < left or value > right:
            continue
        if value == right:
            counts[tid, width - 1] += 1
            continue
        idx = np.searchsorted(edges, value, side="right") - 1
        if 0 <= idx < width:
            counts[tid, idx] += 1

    merged = np.zeros(width, dtype=np.int64)
    finite_count = 0
    for tid in range(n_threads):
        merged += counts[tid]
        finite_count += finite_counts[tid]

    return merged, finite_count


@numba.njit(parallel=True, boundscheck=True)
def _finite_2d_range_numba(scores, intensities, threshold):
    n_threads = numba.get_num_threads()
    score_mins = np.full(n_threads, np.inf, dtype=np.float64)
    score_maxs = np.full(n_threads, -np.inf, dtype=np.float64)
    int_mins = np.full(n_threads, np.inf, dtype=np.float64)
    int_maxs = np.full(n_threads, -np.inf, dtype=np.float64)
    counts = np.zeros(n_threads, dtype=np.int64)

    for i in numba.prange(len(scores)):
        tid = numba.get_thread_id()
        score = scores[i]
        if not np.isfinite(score):
            continue
        log_int = np.log10(1.0 + intensities[i])
        if log_int < threshold:
            continue
        if score < score_mins[tid]:
            score_mins[tid] = score
        if score > score_maxs[tid]:
            score_maxs[tid] = score
        if log_int < int_mins[tid]:
            int_mins[tid] = log_int
        if log_int > int_maxs[tid]:
            int_maxs[tid] = log_int
        counts[tid] += 1

    found = False
    score_min = 0.0
    score_max = 0.0
    int_min = 0.0
    int_max = 0.0
    finite_count = 0
    for tid in range(n_threads):
        if counts[tid] == 0:
            continue
        finite_count += counts[tid]
        if not found:
            score_min = score_mins[tid]
            score_max = score_maxs[tid]
            int_min = int_mins[tid]
            int_max = int_maxs[tid]
            found = True
        else:
            if score_mins[tid] < score_min:
                score_min = score_mins[tid]
            if score_maxs[tid] > score_max:
                score_max = score_maxs[tid]
            if int_mins[tid] < int_min:
                int_min = int_mins[tid]
            if int_maxs[tid] > int_max:
                int_max = int_maxs[tid]

    return found, score_min, score_max, int_min, int_max, finite_count


@numba.njit(parallel=True, boundscheck=True)
def _finite_hist2d_numba(scores, intensities, score_edges, int_edges, threshold):
    n_threads = numba.get_num_threads()
    counts = np.zeros(
        (n_threads, len(int_edges) - 1, len(score_edges) - 1), dtype=np.int64
    )
    finite_counts = np.zeros(n_threads, dtype=np.int64)
    score_left = score_edges[0]
    score_right = score_edges[-1]
    int_left = int_edges[0]
    int_right = int_edges[-1]
    height = counts.shape[1]
    width = counts.shape[2]

    for i in numba.prange(len(scores)):
        tid = numba.get_thread_id()
        score = scores[i]
        if not np.isfinite(score):
            continue
        log_int = np.log10(1.0 + intensities[i])
        if log_int < threshold:
            continue
        finite_counts[tid] += 1
        if score < score_left or score > score_right:
            continue
        if log_int < int_left or log_int > int_right:
            continue

        if score == score_right:
            x = width - 1
        else:
            x = np.searchsorted(score_edges, score, side="right") - 1
        if log_int == int_right:
            y = height - 1
        else:
            y = np.searchsorted(int_edges, log_int, side="right") - 1

        if 0 <= x < width and 0 <= y < height:
            counts[tid, y, x] += 1

    merged = np.zeros((height, width), dtype=np.int64)
    finite_count = 0
    for tid in range(n_threads):
        merged += counts[tid]
        finite_count += finite_counts[tid]

    return merged, finite_count


def _chunk_bounds(length, chunk_size=CHUNK_SIZE):
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def _build_valid_edges(scores, intensities=None, threshold=None, n_bins: int = 100):
    found = False
    score_min = 0.0
    score_max = 0.0
    finite_count = 0

    for start, end in _chunk_bounds(len(scores)):
        score_chunk = np.asarray(scores[start:end])
        if threshold is None:
            chunk_found, chunk_min, chunk_max, chunk_count = _finite_min_max_numba(score_chunk)
        else:
            intensity_chunk = np.asarray(intensities[start:end])
            chunk_found, chunk_min, chunk_max, chunk_count = _finite_min_max_threshold_numba(
                score_chunk, intensity_chunk, threshold
            )
        finite_count += chunk_count
        if not chunk_found:
            continue
        if not found:
            score_min = chunk_min
            score_max = chunk_max
            found = True
        else:
            score_min = min(score_min, chunk_min)
            score_max = max(score_max, chunk_max)

    if not found:
        return None, 0

    score_min, score_max = _expand_if_degenerate(float(score_min), float(score_max))
    return np.linspace(score_min, score_max, n_bins + 1), finite_count


def _chunked_histogram(scores, edges, intensities=None, threshold=None):
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    finite_count = 0

    for start, end in _chunk_bounds(len(scores)):
        score_chunk = np.asarray(scores[start:end])
        if threshold is None:
            chunk_counts, chunk_count = _finite_histogram_numba(score_chunk, edges)
        else:
            intensity_chunk = np.asarray(intensities[start:end])
            chunk_counts, chunk_count = _finite_histogram_threshold_numba(
                score_chunk, intensity_chunk, edges, threshold
            )
        counts += chunk_counts
        finite_count += chunk_count

    return counts, finite_count


def _chunked_2d_histogram(scores, intensities, threshold=None, bins_2d=BINS_2D):
    threshold = -1.0 if threshold is None else float(threshold)
    found = False
    score_min = 0.0
    score_max = 0.0
    int_min = 0.0
    int_max = 0.0
    finite_count = 0

    for start, end in _chunk_bounds(len(scores)):
        score_chunk = np.asarray(scores[start:end])
        intensity_chunk = np.asarray(intensities[start:end])
        chunk_found, c_score_min, c_score_max, c_int_min, c_int_max, chunk_count = _finite_2d_range_numba(
            score_chunk, intensity_chunk, threshold
        )
        finite_count += chunk_count
        if not chunk_found:
            continue
        if not found:
            score_min, score_max = c_score_min, c_score_max
            int_min, int_max = c_int_min, c_int_max
            found = True
        else:
            score_min = min(score_min, c_score_min)
            score_max = max(score_max, c_score_max)
            int_min = min(int_min, c_int_min)
            int_max = max(int_max, c_int_max)

    if not found:
        return None, None, 0, None, None

    score_min, score_max = _expand_if_degenerate(float(score_min), float(score_max))
    int_min, int_max = _expand_if_degenerate(float(int_min), float(int_max))
    score_edges = np.linspace(score_min, score_max, bins_2d[1] + 1)
    int_edges = np.linspace(int_min, int_max, bins_2d[0] + 1)
    hist = np.zeros((bins_2d[0], bins_2d[1]), dtype=np.int64)

    for start, end in _chunk_bounds(len(scores)):
        score_chunk = np.asarray(scores[start:end])
        intensity_chunk = np.asarray(intensities[start:end])
        chunk_hist, _ = _finite_hist2d_numba(
            score_chunk, intensity_chunk, score_edges, int_edges, threshold
        )
        hist += chunk_hist

    extent = ((int_min, int_max), (score_min, score_max))
    return hist, extent, finite_count, score_edges, int_edges


def main():
    p = argparse.ArgumentParser(
        description="Map Sage scores against pmsms mapping results."
    )
    p.add_argument("precursors", type=Path, help="Filtered precursors parquet.")
    p.add_argument("pmsms", type=Path, help="pmsms mmappet dataset directory.")
    p.add_argument(
        "mapped_precursors", type=Path, help="Mapping output precursors.parquet."
    )
    p.add_argument("mapping", type=Path, help="Mapping output mapping.parquet.")
    p.add_argument("--config", type=Path, required=True, help="Pipeline config TOML.")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output directory.")
    args = p.parse_args()
    compare_scores(**args.__dict__)


try:
    in_ipython = get_ipython() is not None  # type: ignore[name-defined]
except NameError:
    in_ipython = False

if __name__ == "__main__":
    from pprint import pprint

    dataset = "F9468"
    dataset = "F9477"
    cfg = "optimal2tier4"
    sage_version = "devel_fixed"
    sage_cfg = "p12f15"
    fasta = "human"

    __args = dict(
        precursors=f"temp/{dataset}/{cfg}/filtered_precursor_clusters_with_nontrivial_ms2.parquet",
        pmsms=f"temp/{dataset}/{cfg}/pmsms.mmappet",
        mapped_precursors=f"temp/{dataset}/{cfg}/sage/{sage_version}/{sage_cfg}/{fasta}/results/sage_mapped_to_pmsms/precursors.parquet",
        mapping=f"temp/{dataset}/{cfg}/sage/{sage_version}/{sage_cfg}/{fasta}/results/sage_mapped_to_pmsms/mapping.parquet",
        config=f"configs/{cfg}.toml",
        output=f"/home/matteo/temp/{dataset}_{cfg}_mappedback_scores",
    )
    pprint(__args)
    locals().update(**__args)
    filter_label = ""
    score_label = ""


# ── Individual plot functions (each saves one file) ───────────────────────────


def _plot_score_distribution(
    output: Path,
    edges,
    all_counts,
    matched_counts,
    score_label: str,
    filter_label: str = "",
):
    unmatched_counts = all_counts - matched_counts
    n_matched = int(matched_counts.sum())
    n_unmatched = int(unmatched_counts.sum())
    title_suffix = f"\n{filter_label}" if filter_label else ""

    def _density(counts):
        widths = np.diff(edges)
        total = counts.sum()
        if total <= 0:
            return np.zeros_like(counts, dtype=np.float64)
        return counts / (total * widths)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stairs(
        _density(matched_counts),
        edges,
        fill=True,
        alpha=0.5,
        label=f"matched ({n_matched:_})",
    )
    ax.stairs(
        _density(unmatched_counts),
        edges,
        fill=True,
        alpha=0.5,
        label=f"unmatched ({n_unmatched:_})",
    )
    ax.legend()
    ax.set_xlabel("pmsms score")
    ax.set_ylabel("density")
    ax.set_title(
        f"{score_label}\n({n_matched:_} matched / {n_unmatched:_} unmatched){title_suffix}",
        fontsize=9,
    )
    fig.tight_layout()
    if in_ipython:
        plt.show()
    else:
        fig.savefig(output / "score_distribution.png", dpi=150)
        plt.close(fig)


def _plot_score_by_charge(
    output: Path,
    edges,
    matched_scores,
    charge_per_fragment,
    unmatched_counts,
    score_label: str,
    filter_label: str = "",
):
    title_suffix = f"\n{filter_label}" if filter_label else ""
    n_unmatched = int(unmatched_counts.sum())

    matched_plot_df = pd.DataFrame(
        {
            "score": matched_scores,
            "charge": charge_per_fragment.astype(str),
        }
    )

    bin_centers = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(0)
    if n_unmatched > 0 and unmatched_counts.sum() > 0:
        unmatched_sample = rng.choice(
            bin_centers,
            size=min(200_000, n_unmatched),
            p=unmatched_counts / unmatched_counts.sum(),
        )
    else:
        unmatched_sample = np.array([], dtype=np.float64)
    unmatched_plot_df = pd.DataFrame({"score": unmatched_sample})

    p = (
        P.ggplot(matched_plot_df, P.aes("score"))
        + P.geom_histogram(
            P.aes(y=P.after_stat("density")), bins=50, fill="steelblue", alpha=0.6
        )
        + P.geom_histogram(
            unmatched_plot_df,
            P.aes("score", y=P.after_stat("density")),
            bins=50,
            fill="#808080",
            alpha=0.4,
        )
        + P.facet_wrap("~charge")
        + P.labs(
            x="pmsms score",
            y="density",
            title=f"{score_label}\nblue=matched by charge, grey=unmatched (all facets){title_suffix}",
        )
        + P.theme_bw()
        + P.theme(plot_title=P.element_text(size=7))
    )
    if in_ipython:
        p.show()
    else:
        p.save(
            output / "score_by_charge.png", dpi=150, verbose=False, width=14, height=6
        )


def _plot_score_vs_intensity_2d(
    output: Path,
    h_matched,
    h_unmatched,
    extent,
    score_label: str,
    filter_label: str = "",
):
    title_suffix = f"\n{filter_label}" if filter_label else ""

    def _norm2d(h):
        total = h.sum()
        if total <= 0:
            return np.zeros_like(h, dtype=np.float64)
        return h / total

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    imshow_kw = dict(
        origin="lower",
        aspect="auto",
        cmap="inferno",
        extent=[extent[0][0], extent[0][1], extent[1][0], extent[1][1]],
    )
    for ax, h, label in zip(axes, [h_matched, h_unmatched], ["matched", "unmatched"]):
        ax.imshow(_norm2d(h).T, **imshow_kw)
        ax.set_xlabel("log10(intensity+1)")
        ax.set_ylabel("pmsms score")
        ax.set_title(label)
    fig.suptitle(f"{score_label}{title_suffix}", fontsize=9)
    fig.tight_layout()
    if in_ipython:
        plt.show()
    else:
        fig.savefig(output / "score_vs_intensity_2d.png", dpi=150)
        plt.close(fig)


def _plot_score_vs_intensity_isoquants(
    output: Path,
    h_matched,
    h_unmatched,
    extent,
    bins_2d,
    score_label: str,
    filter_label: str = "",
):
    title_suffix = f"\n{filter_label}" if filter_label else ""
    x_centers = np.linspace(extent[0][0], extent[0][1], bins_2d[0])
    y_centers = np.linspace(extent[1][0], extent[1][1], bins_2d[1])

    def _norm2d(h):
        total = h.sum()
        if total <= 0:
            return np.zeros_like(h, dtype=np.float64)
        return h / total

    def _smooth(h, sigma=2.0):
        return gaussian_filter(_norm2d(h).astype(np.float64), sigma=sigma)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(8, 5))
        levels = 10
        ax.contour(
            x_centers,
            y_centers,
            _smooth(h_matched).T,
            levels=levels,
            colors="steelblue",
        )
        ax.contour(
            x_centers,
            y_centers,
            _smooth(h_unmatched).T,
            levels=levels,
            colors="darkorange",
        )
        ax.legend(
            [Line2D([0], [0], color="steelblue"), Line2D([0], [0], color="darkorange")],
            ["matched", "unmatched"],
        )
        ax.set_xlabel("log10(intensity+1)")
        ax.set_ylabel("pmsms score")
        ax.set_title(f"{score_label}{title_suffix}", fontsize=9)
        fig.tight_layout()
    if in_ipython:
        plt.show()
    else:
        fig.savefig(output / "score_vs_intensity_isoquants.png", dpi=150)
        plt.close(fig)


def _make_plots(
    output: Path,
    pmsms_score,
    pmsms_intensity,
    matched_scores,
    matched_intensity,
    charge_per_fragment,
    score_label: str,
    filter_label: str = "",
    intensity_threshold: float | None = None,
):
    """Pre-compute shared histograms and return a list of (fn, kwargs) plot tasks."""
    Path(output).mkdir(parents=True, exist_ok=True)

    matched_scores_arr = np.asarray(matched_scores, dtype=np.float64)
    matched_intensity_arr = np.asarray(matched_intensity, dtype=np.uint32)
    charge_per_fragment_arr = np.asarray(charge_per_fragment)

    log10_int_matched_arr = np.log10(1 + matched_intensity_arr.astype(np.float32))
    matched_mask = (
        np.isfinite(matched_scores_arr)
        & np.isfinite(log10_int_matched_arr)
        & np.asarray(pd.notna(charge_per_fragment_arr))
    )
    if intensity_threshold is not None:
        matched_mask &= log10_int_matched_arr >= intensity_threshold

    matched_scores_plot = matched_scores_arr[matched_mask]
    log10_int_matched_plot = log10_int_matched_arr[matched_mask]
    matched_intensity_plot = matched_intensity_arr[matched_mask]
    charge_per_fragment_plot = charge_per_fragment_arr[matched_mask]
    _describe_drop(f"{output}/matched", len(matched_scores_arr), len(matched_scores_plot))

    h_all, extent, finite_all_2d, score_edges, _int_edges = _chunked_2d_histogram(
        pmsms_score, pmsms_intensity, intensity_threshold
    )
    if h_all is None or extent is None:
        print(f"{output}: no finite score/intensity pairs; skipping 2D plots.")
        return []
    _describe_drop(f"{output}/2d", len(pmsms_score), finite_all_2d)

    h_matched_2d = kg.histogram2D(
        log10_int_matched_plot, matched_scores_plot, extent=extent, bins=BINS_2D
    )
    h_unmatched_2d = h_all - h_matched_2d
    edges = score_edges
    all_counts = h_all.sum(axis=0)
    matched_counts = h_matched_2d.sum(axis=0)
    unmatched_counts = h_unmatched_2d.sum(axis=0)

    common = dict(score_label=score_label, filter_label=filter_label)
    return [
        (
            _plot_score_distribution,
            dict(
                output=output,
                edges=edges,
                all_counts=all_counts,
                matched_counts=matched_counts,
                **common,
            ),
        ),
        (
            _plot_score_by_charge,
            dict(
                output=output,
                edges=edges,
                matched_scores=matched_scores_plot,
                charge_per_fragment=charge_per_fragment_plot,
                unmatched_counts=unmatched_counts,
                **common,
            ),
        ),
        (
            _plot_score_vs_intensity_2d,
            dict(
                output=output,
                h_matched=h_matched_2d,
                h_unmatched=h_unmatched_2d,
                extent=extent,
                **common,
            ),
        ),
        (
            _plot_score_vs_intensity_isoquants,
            dict(
                output=output,
                h_matched=h_matched_2d,
                h_unmatched=h_unmatched_2d,
                extent=extent,
                bins_2d=BINS_2D,
                **common,
            ),
        ),
    ]


def compare_scores(
    precursors: str | Path,
    pmsms: str | Path,
    mapped_precursors: str | Path,
    mapping: str | Path,
    config: str | Path,
    output: str | Path,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # ── 1. Open files ─────────────────────────────────────────────────────────
    with open(config, "rb") as fh:
        cfg = tomllib.load(fh)
    pmsms_cfg = cfg["pseudomsms"]
    score_method = pmsms_cfg["tofs_extraction_method"]
    if score_method == "score":
        params = pmsms_cfg.get("tofs_extraction_params", {})
        params_str = " | ".join(f"{k}={v}" for k, v in params.items())
        score_label = textwrap.fill(f"score | {params_str}", width=80)
    else:
        score_label = score_method

    pmsms_data = mmappet.open_dataset_dct(pmsms)
    pmsms_score = pmsms_data["score"]
    pmsms_intensity = pmsms_data["intensity"]
    mapping_df = pd.read_parquet(mapping)
    mapped_prec_df = pd.read_parquet(mapped_precursors)

    # ── 2. Extract pmsms scores / intensities for matched fragments ───────────
    matched_idx = mapping_df["pmsms_fragment_idx"].to_numpy()
    matched_scores = pmsms_score[matched_idx]
    matched_intensity = pmsms_intensity[matched_idx]

    charge_per_fragment = np.repeat(
        mapped_prec_df["detected_charges"].to_numpy(),
        mapped_prec_df["mapped_cnt"].to_numpy(),
    )

    # ── 3. Build all plot tasks (all + q30 filter levels) ────────────────────
    tasks = _make_plots(
        output / "all",
        pmsms_score,
        pmsms_intensity,
        matched_scores,
        matched_intensity,
        charge_per_fragment,
        score_label,
    )
    log10_int_matched = np.log10(1 + np.asarray(matched_intensity, dtype=np.float32))
    finite_q30_matched = np.isfinite(matched_scores) & np.isfinite(log10_int_matched)
    if not finite_q30_matched.any():
        print("q30: no finite matched rows; skipping q30 plots.")
    else:
        threshold = float(np.quantile(log10_int_matched[finite_q30_matched], 0.30))
        tasks += _make_plots(
            output / "q30",
            pmsms_score,
            pmsms_intensity,
            matched_scores,
            matched_intensity,
            charge_per_fragment,
            score_label,
            filter_label=f"log10(intensity+1) ≥ {threshold:.3f} (q30 of matched)",
            intensity_threshold=threshold,
        )

    # ── 4. Run plots sequentially to avoid large process copies ───────────────
    if not tasks:
        print("   No plot tasks to run.")
        return
    for fn, kwargs in tasks:
        fn(**kwargs)

    print("   Done!")
