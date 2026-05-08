import argparse
from pathlib import Path

import mmappet
import numba
import numpy as np
import pandas as pd

from searchops.plotting import make_kebab_df, kebab_plot


@numba.njit(parallel=True)
def _log101p(xx):
    return np.log10(xx + 1)


def _gather_frags(di_sample, frags_df):
    starts = di_sample["idx"].to_numpy(dtype=np.int64)
    sizes = di_sample["size"].to_numpy(dtype=np.int64)
    idx = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, sizes)])
    idx.sort()
    df = frags_df.iloc[idx].reset_index(drop=True)
    df["pmsms_fragment_idx"] = idx
    return df


def make_kebab_plots(
    pmsms: Path,
    neighbor_score: Path,
    fdr_parquet: Path,
    mapping: Path,
    output: Path,
    n_prec: int = 1_000,
    n_bins: int = 100,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # ── 1. Load fragments + neighbor_score ────────────────────────────────────
    frags = mmappet.open_dataset(pmsms)
    nb = mmappet.open_dataset(neighbor_score)
    frags["neighbor_score"] = nb["neighbor_score"].to_numpy()
    frags["score_delta"] = frags["neighbor_score"] - frags["score"]
    frags["log10intensity"] = _log101p(frags["intensity"].to_numpy())

    # ── 2. Find SAGE-found precursor indices ──────────────────────────────────
    sage_fdr = pd.read_parquet(fdr_parquet, columns=["scannr"])
    found_idx = set(
        sage_fdr["scannr"].str.extract(r"precursor_idx=(\d+)")[0].astype(np.int64).tolist()
    )
    print(f"SAGE-found precursors: {len(found_idx):_}")

    dataindex = mmappet.open_dataset(Path(pmsms) / "dataindex.mmappet")
    mask_found = dataindex["precursor_idx"].isin(found_idx)
    di_found = dataindex[mask_found].reset_index(drop=True)
    print(f"Precursors in dataindex — found: {len(di_found):_}")

    # ── 3. Sample found precursors, gather their fragments ───────────────────
    rng = np.random.default_rng(42)
    s_found = di_found.iloc[
        rng.choice(len(di_found), size=min(n_prec, len(di_found)), replace=False)
    ]
    ff = _gather_frags(s_found, frags)

    # ── 4. Split matched / unmatched within found precursors ──────────────────
    matched_idx = set(
        pd.read_parquet(Path(mapping) / "mapping.parquet", columns=["pmsms_fragment_idx"])[
            "pmsms_fragment_idx"
        ].tolist()
    )
    ff_matched_mask = ff["pmsms_fragment_idx"].isin(matched_idx)
    fm = ff[ff_matched_mask].dropna().reset_index(drop=True)
    fu = ff[~ff_matched_mask].dropna().reset_index(drop=True)
    print(f"Matched: {len(fm):_}, Unmatched: {len(fu):_}")

    # ── 5. Generate 5 kebab plots ─────────────────────────────────────────────
    plots = [
        (
            fm["score"].to_numpy(),           fm["score_delta"].to_numpy(),
            fu["score"].to_numpy(),           fu["score_delta"].to_numpy(),
            "score bin", "score_delta", "score_delta | score bin",
            "score_delta_vs_score.png",
        ),
        (
            fm["log10intensity"].to_numpy(),   fm["score_delta"].to_numpy(),
            fu["log10intensity"].to_numpy(),   fu["score_delta"].to_numpy(),
            "log10(1+intensity) bin", "score_delta", "score_delta | log10(1+intensity) bin",
            "score_delta_vs_log10intensity.png",
        ),
        (
            fm["log10intensity"].to_numpy(),   fm["score"].to_numpy(),
            fu["log10intensity"].to_numpy(),   fu["score"].to_numpy(),
            "log10(1+intensity) bin", "score", "score | log10(1+intensity) bin",
            "score_vs_log10intensity.png",
        ),
        (
            fm["tof"].to_numpy(dtype=np.float64), fm["score"].to_numpy(),
            fu["tof"].to_numpy(dtype=np.float64), fu["score"].to_numpy(),
            "tof bin", "score", "score | tof bin",
            "score_vs_tof.png",
        ),
        (
            fm["tof"].to_numpy(dtype=np.float64), fm["score_delta"].to_numpy(),
            fu["tof"].to_numpy(dtype=np.float64), fu["score_delta"].to_numpy(),
            "tof bin", "score_delta", "score_delta | tof bin",
            "score_delta_vs_tof.png",
        ),
    ]

    for x_m, y_m, x_u, y_u, x_label, y_label, title, fname in plots:
        df = make_kebab_df(x_m, y_m, x_u, y_u, n_bins=n_bins)
        p = kebab_plot(df, x_label=x_label, y_label=y_label, title=title)
        p.save(output / fname, dpi=150, verbose=False, width=12, height=20)
        print(f"  saved {output / fname}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Score-delta kebab plots: SAGE-matched vs unmatched fragments."
    )
    parser.add_argument("pmsms",          type=Path, help="pmsms mmappet directory")
    parser.add_argument("neighbor_score", type=Path, help="neighbor_score mmappet directory")
    parser.add_argument("fdr_parquet",    type=Path, help="SAGE FDR-filtered precursors parquet")
    parser.add_argument("mapping",        type=Path, help="sage_mapped_to_pmsms directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-prec", type=int, default=1_000, metavar="N",
                        help="Precursors to sample from found group (default: 1000)")
    parser.add_argument("--n-bins", type=int, default=100, metavar="N",
                        help="Bins per kebab plot (default: 100)")
    args = parser.parse_args()
    make_kebab_plots(
        pmsms=args.pmsms,
        neighbor_score=args.neighbor_score,
        fdr_parquet=args.fdr_parquet,
        mapping=args.mapping,
        output=args.output,
        n_prec=args.n_prec,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    from pprint import pprint

    dataset = "F9477"
    cfg = "optimal2tier"
    sage_version = "devel_fixed"
    sage_cfg = "p12f15"
    fasta = "human"
    mgf_branch = "tof_filter/default/default"

    __args = dict(
        pmsms=f"temp/{dataset}/{cfg}/pmsms.mmappet",
        neighbor_score=f"temp/{dataset}/{cfg}/{mgf_branch}/neighbor_score.mmappet",
        fdr_parquet=f"temp/{dataset}/{cfg}/{mgf_branch}/sage/{sage_version}/{sage_cfg}/{fasta}/results/results.sage.fdr.parquet",
        mapping=f"temp/{dataset}/{cfg}/{mgf_branch}/sage/{sage_version}/{sage_cfg}/{fasta}/results/sage_mapped_to_pmsms",
        output=f"/home/matteo/temp/{dataset}_{cfg}_score_delta_kebab",
        n_prec=1_000,
        n_bins=100,
    )
    pprint(__args)
    make_kebab_plots(**__args)
