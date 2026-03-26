import argparse
import textwrap
import tomllib
from pathlib import Path

import plotnine as P
import mmappet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kilograms as kg

from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


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


if __name__ == "__main__":
    try:
        in_ipython = get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        in_ipython = False
    if in_ipython:
        plt.show()
    else:
        fig.savefig(output / "score_distribution.png", dpi=150)
        plt.close(fig)

    from pprint import pprint

    dataset = "F9477"
    dataset = "F9468"
    cfg = "optimal2tier"
    sage_version = "devel"
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


def _make_plots(
    output: Path,
    pmsms_score,
    log10_int_all,
    matched_scores,
    log10_int_matched,
    charge_per_fragment,
    score_label: str,
    filter_label: str = "",
):
    Path(output).mkdir(parents=True, exist_ok=True)
    title_suffix = f"\n{filter_label}" if filter_label else ""

    # ── 3. Plot histogram ─────────────────────────────────────────────────────
    score_min = float(pmsms_score.min())
    score_max = float(pmsms_score.max())
    bins = np.linspace(score_min, score_max, 101)

    all_counts, edges = np.histogram(pmsms_score, bins=bins)
    matched_counts, _ = np.histogram(matched_scores, bins=bins)
    unmatched_counts = all_counts - matched_counts

    def _density(counts):
        widths = np.diff(edges)
        return counts / (counts.sum() * widths)

    n_matched = int(matched_counts.sum())
    n_unmatched = int(unmatched_counts.sum())

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

    # ── 4. per-charge conditional histograms ────────────────────────

    matched_plot_df = pd.DataFrame(
        {
            "score": matched_scores,
            "charge": charge_per_fragment.astype(str),
        }
    )

    # Reconstruct an unmatched reference sample from the pre-computed histogram
    # (avoids materialising a ~187M array).  No 'charge' column → appears in all facets.
    bin_centers = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(0)
    unmatched_sample = rng.choice(
        bin_centers,
        size=min(200_000, n_unmatched),
        p=unmatched_counts / unmatched_counts.sum(),
    )
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

    # ── 5. score vs log10(intensity), matched vs unmatched ──────

    extent = (
        (float(log10_int_all.min()), float(log10_int_all.max())),
        (float(pmsms_score.min()), float(pmsms_score.max())),
    )
    bins_2d = (200, 200)
    h_all = kg.histogram2D(log10_int_all, pmsms_score, extent=extent, bins=bins_2d)
    h_matched = kg.histogram2D(
        log10_int_matched, matched_scores, extent=extent, bins=bins_2d
    )
    h_unmatched = h_all - h_matched

    def _norm2d(h):
        return h / h.sum()

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

    # ── 6. Isoquant overlay ───────────────────────────────────────────────────

    x_centers = np.linspace(extent[0][0], extent[0][1], bins_2d[0])
    y_centers = np.linspace(extent[1][0], extent[1][1], bins_2d[1])

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

    log10_int_all = np.log10(1 + pmsms_intensity.astype(np.float32))
    log10_int_matched = log10_int_all[matched_idx]

    # charge label per matched fragment via CSR expansion
    charge_per_fragment = np.repeat(
        mapped_prec_df["detected_charges"].to_numpy(),
        mapped_prec_df["mapped_cnt"].to_numpy(),
    )

    # ── 3-6. Plots: all events ────────────────────────────────────────────────
    _make_plots(
        output / "all",
        pmsms_score,
        log10_int_all,
        matched_scores,
        log10_int_matched,
        charge_per_fragment,
        score_label,
    )

    # ── 3-6. Plots: events above 30th-percentile of matched intensities ───────
    threshold = float(np.quantile(log10_int_matched, 0.30))
    above_all = log10_int_all >= threshold
    above_matched = log10_int_matched >= threshold

    _make_plots(
        output / "q30",
        pmsms_score[above_all],
        log10_int_all[above_all],
        matched_scores[above_matched],
        log10_int_matched[above_matched],
        charge_per_fragment[above_matched],
        score_label,
        filter_label=f"log10(intensity+1) ≥ {threshold:.3f} (q30 of matched)",
    )
