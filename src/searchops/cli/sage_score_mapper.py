import argparse
import textwrap
import tomllib
from pathlib import Path

import mmappet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    pmsms_score = mmappet.open_dataset_dct(pmsms)["score"]
    mapping_df = pd.read_parquet(mapping)
    mapped_prec_df = pd.read_parquet(mapped_precursors)  # noqa: F841  reserved

    # ── 2. Extract pmsms scores for matched fragments ─────────────────────────
    matched_idx = mapping_df["pmsms_fragment_idx"].to_numpy()
    matched_scores = pmsms_score[matched_idx]

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
    ax.stairs(_density(matched_counts), edges, fill=True, alpha=0.5,
              label=f"matched ({n_matched:_})")
    ax.stairs(_density(unmatched_counts), edges, fill=True, alpha=0.5,
              label=f"unmatched ({n_unmatched:_})")
    ax.legend()
    ax.set_xlabel("pmsms score")
    ax.set_ylabel("density")
    ax.set_title(f"{score_label}\n({n_matched:_} matched / {n_unmatched:_} unmatched)", fontsize=9)
    fig.tight_layout()
    try:
        in_ipython = get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        in_ipython = False
    if in_ipython:
        plt.show()
    else:
        fig.savefig(output / "score_distribution.png", dpi=150)
        plt.close(fig)
