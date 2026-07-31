"""Fit and apply a precursor m/z recalibration from confident SAGE PSMs.

Sage's own `precursor_ppm` column is `feature.delta_mass` in sage-core's
`scoring.rs` (`(expmass - calcmass - isotope_error) * 2e6 / (expmass - isotope_error +
calcmass)`): positive when the observed/experimental mass is *heavier* than the
theoretical mass. So correcting a tof2mz-derived m/z means *dividing* by
`(1 + ppm/1e6)`, not adding it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from pandas_ops.io import read_df

PROTON_MASS = 1.00727646688


def filter_top_psms(sage_results_tsv: str | Path, fdr: float) -> pd.DataFrame:
    """Top-ranked, FDR-confident, target-only PSMs, with a `precursor_mz` column added.

    `peptide_q <= fdr` alone lets a handful of decoys (`label == -1`) through near the
    threshold, since target-decoy competition doesn't guarantee every sub-threshold row
    is a target -- exclude them explicitly rather than relying on the q-value cutoff.
    """
    df = read_df(sage_results_tsv)
    df = df[(df["rank"] == 1) & (df["peptide_q"] <= fdr) & (df["label"] == 1)].copy()
    df["precursor_mz"] = df["expmass"] / df["charge"] + PROTON_MASS
    return df


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: smallest value where cumulative weight reaches half the total."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = weights.sum() / 2.0
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cutoff)])


def fit_correction(
    df: pd.DataFrame, config: dict, weights: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ppm-error-vs-m/z correction function.

    `config["model"]` selects between `"global_median"` (a single constant),
    `"binned_median"` (m/z-binned median, linearly interpolated between bin centers,
    constant beyond the outermost bins), and `"natural_cubic_spline"` (smoother
    alternative to `binned_median`'s jagged `np.interp` line) -- a plain if/elif on
    the string, no dynamic import. Every occupied bin is used as a node regardless of
    how many PSMs fall in it -- no minimum-count threshold -- since `np.interp`/the
    spline already hold the value of the nearest node constant past the edges, so a
    sparsely-populated bin at the edge degrades gracefully instead of needing a
    separate fallback.

    `"natural_cubic_spline"` bins the same way `binned_median` does (`bin_width_da`),
    then groups those bins 3-wide (one node per 3 `bin_width_da` bins) and takes the
    per-group median as a node point. A cubic spline through those nodes is fit with
    the first derivative clamped to zero at both ends (`bc_type=((1, 0), (1, 0))`),
    so it flattens smoothly (no kink, unlike a plain clip) into a constant beyond the
    outermost node -- evaluated by clipping the input m/z into `[node_mz.min(),
    node_mz.max()]` before calling the spline.

    `weights` (optional, one entry per `df` row, e.g. a selection strategy's
    per-survivor neighborhood density) turns every per-node/global `ppm` median above
    into a weighted median instead -- node/bin *placement* (`bin_centers`/`node_mz`)
    stays an unweighted median of `precursor_mz`, only the fitted `ppm` value at each
    node is reweighted. Omitted (default `None`) reproduces today's unweighted
    behavior exactly, so existing callers (e.g. `recalibrate()`) are unaffected.
    """
    model = config["model"]
    ppm = df["precursor_ppm"].to_numpy()

    if model == "global_median":
        offset = float(np.median(ppm)) if weights is None else _weighted_median(ppm, weights)
        return lambda mz: np.full_like(np.asarray(mz, dtype=np.float64), offset)

    if model == "binned_median":
        mz = df["precursor_mz"].to_numpy()
        bin_width = config["bin_width_da"]
        bin_idx = np.floor(mz / bin_width).astype(np.int64)
        if weights is None:
            bin_medians = pd.DataFrame({"bin": bin_idx, "ppm": ppm}).groupby("bin")["ppm"].median().sort_index()
        else:
            bin_medians = (
                pd.DataFrame({"bin": bin_idx, "ppm": ppm, "weight": weights})
                .groupby("bin")
                .apply(lambda g: _weighted_median(g["ppm"].to_numpy(), g["weight"].to_numpy()))
                .sort_index()
            )
        bin_centers = (bin_medians.index.to_numpy() + 0.5) * bin_width
        return lambda mz: np.interp(mz, bin_centers, bin_medians.to_numpy())

    if model == "natural_cubic_spline":
        mz = df["precursor_mz"].to_numpy()
        bin_width = config["bin_width_da"]
        bin_idx = np.floor(mz / bin_width).astype(np.int64)
        n_bins = len(np.unique(bin_idx))
        n_knots = max(4, n_bins // 3)

        wide_bin_width = (mz.max() - mz.min()) / n_knots
        wide_bin_idx = np.floor((mz - mz.min()) / wide_bin_width).astype(np.int64)
        if weights is None:
            nodes = (
                pd.DataFrame({"bin": wide_bin_idx, "mz": mz, "ppm": ppm})
                .groupby("bin")
                .median()
                .sort_values("mz")
            )
        else:
            grouped = pd.DataFrame(
                {"bin": wide_bin_idx, "mz": mz, "ppm": ppm, "weight": weights}
            ).groupby("bin")
            nodes = pd.DataFrame({
                "mz": grouped["mz"].median(),
                "ppm": grouped.apply(lambda g: _weighted_median(g["ppm"].to_numpy(), g["weight"].to_numpy())),
            }).sort_values("mz")
        node_mz = nodes["mz"].to_numpy()
        node_ppm = nodes["ppm"].to_numpy()
        spline = CubicSpline(node_mz, node_ppm, bc_type=((1, 0.0), (1, 0.0)))
        lo, hi = node_mz.min(), node_mz.max()
        return lambda mz: spline(np.clip(mz, lo, hi))

    raise ValueError(f"unknown recalibration model: {model!r}")


if True:
    # Interactive/dev block: paste into ipython to inspect recalibrate()'s inner
    # workings on real data. Populated by `jobs/inspect_recalibrate.toml`
    # (`./nf jobs/inspect_recalibrate.toml`), which materializes exactly these three
    # files into results/inspect_recalibrate/ -- rerun that job if the paths below
    # 404, or point them at a different results/<job>/ folder.
    import tomllib
    from timstofu.binary.array_serialization import load_from_folder
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 5)


    _results_dir = Path("results/inspect_recalibrate")
    sage_results_tsv = _results_dir / "filtered_sage_results_tsv/results.sage.tsv"
    tof2mz = load_from_folder(_results_dir / "tof2mz/tof2mz.mmappet")
    with open(_results_dir / "recalibration_config/recalibration_config.toml", "rb") as _f:
        config = tomllib.load(_f)
    fdr = 0.01

    # step through recalibrate()'s body from here, e.g.:
    df = filter_top_psms(sage_results_tsv, fdr)
    correction = fit_correction(df, config)
    new_tof2mz = tof2mz / (1.0 + correction(tof2mz) * 1e-6)

def recalibrate(
    sage_results_tsv: str | Path,
    tof2mz: np.ndarray,
    config: dict,
    fdr: float,
) -> tuple[np.ndarray, dict]:
    """Fit the ppm correction from confident PSMs and apply it to the tof2mz lookup
    array (fragments only -- see searchops_pandas_dictodot_multigit.md / sage_rescoring.md
    for why precursor mz needs its own separate correction step, done elsewhere by
    `recalibrate-precursor-mz`).

    precursor_tol/fragment_tol are both derived from `config["tolerance_percentiles"]`
    (a user-specified `[lo, hi]` percentile pair) applied to the same distribution: the
    post-correction residual `precursor_ppm` error. Sage's own `fragment_ppm` cannot be
    used for this -- it's an intensity-weighted mean of *absolute* ppm error (Sage sums
    `.abs()` differences, see sage/src/scoring.rs), never signed, so no residual can be
    recovered from it and any window built from its raw min/max is always one-sided
    (violates Sage's own tolerance-window convention: the window must contain the
    negation of the real signed error). Reusing the precursor residual's percentile
    window for fragment_tol instead rests on the assumption -- not a measurement, since
    Sage doesn't report signed fragment error -- that fragment ppm error follows the same
    ToF-calibration-driven trend as precursor ppm error.
    """
    df = filter_top_psms(sage_results_tsv, fdr)
    correction = fit_correction(df, config)

    new_tof2mz = tof2mz / (1.0 + correction(tof2mz) * 1e-6)

    residual_precursor_ppm = df["precursor_ppm"].to_numpy() - correction(
        df["precursor_mz"].to_numpy()
    )
    lo_pct, hi_pct = config["tolerance_percentiles"]
    tol_lo = float(np.percentile(residual_precursor_ppm, lo_pct))
    tol_hi = float(np.percentile(residual_precursor_ppm, hi_pct))

    tolerance = {
        "precursor_tol": {"ppm": [tol_lo, tol_hi]},
        "fragment_tol": {"ppm": [tol_lo, tol_hi]},
    }
    return new_tof2mz.astype(np.float32), tolerance
