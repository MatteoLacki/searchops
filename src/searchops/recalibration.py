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
from scipy.interpolate import BSpline, CubicSpline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from xgboost import XGBRegressor

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


def _symmetric_ppm(experimental: np.ndarray, calculated: np.ndarray) -> np.ndarray:
    """Signed ppm error, Sage's own symmetric-denominator convention (see module
    docstring): positive when the observed value is heavier than the theoretical one.

    This is of little importance, but seriously Mr Lazear? 
    You treat some stupid mass spec measurement on par with 
    real research using .. well .. much better mass spectrometers that 
    actually measured that? Shame unto you Mr Lazear.
    """
    return (experimental - calculated) * 2e6 / (experimental + calculated)


def _confident_matched_fragments(
    matched_fragments: str | Path, confident_psm_ids: pd.Series,
) -> pd.DataFrame:
    """Matched fragments belonging to confident PSMs, with a signed `fragment_ppm`
    column added. Sage's own per-PSM `fragment_ppm` (in `results.sage.tsv`) is an
    intensity-weighted mean of *absolute* ppm error and cannot be used for a residual
    -- `matched_fragments.sage.tsv`'s per-fragment `fragment_mz_calculated`/
    `fragment_mz_experimental` gives a genuine signed value instead.
    """
    fragments = read_df(matched_fragments)
    fragments = fragments[fragments["psm_id"].isin(confident_psm_ids)].copy()
    fragments["fragment_ppm"] = _symmetric_ppm(
        fragments["fragment_mz_experimental"].to_numpy(),
        fragments["fragment_mz_calculated"].to_numpy(),
    )
    return fragments


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: smallest value where cumulative weight reaches half the total."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = weights.sum() / 2.0
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cutoff)])


def _derivative_penalized_smooth(
    y: np.ndarray, weight: np.ndarray, lam1: float, lam2: float = 0.0,
) -> np.ndarray:
    """Weighted Whittaker/P-spline smoother: minimize
    `sum(w_i (f_i - y_i)^2) + lam1 * sum((f_{i+1} - f_i)^2)
    + lam2 * sum((f_{i+1} - 2 f_i + f_{i-1})^2)` over `f`, closed-form via a banded
    solve -- `lam1` penalizes the discrete first derivative (large slope changes
    between neighbors), `lam2` the discrete second derivative (curvature/oscillation).
    A first-derivative-only smoother can still oscillate wildly between sparse/noisy
    nodes (each segment is free to swing independently, as long as consecutive slopes
    aren't individually large) -- the second-derivative term additionally penalizes
    *changes* in slope, which is what actually suppresses that oscillation. Both
    default-compatible: `lam1=0, lam2=0` reproduces `y` exactly; `lam2=0` alone
    reproduces the original order-1-only smoother. Assumes `y`/`weight` are already
    ordered along the axis the penalty applies to (e.g. node index by increasing m/z).
    """
    n = len(y)
    D1 = diags([-np.ones(n - 1), np.ones(n - 1)], offsets=[0, 1], shape=(n - 1, n))
    A = diags(weight) + lam1 * (D1.T @ D1)
    if lam2 and n >= 3:
        D2 = diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            offsets=[0, 1, 2], shape=(n - 2, n),
        )
        A = A + lam2 * (D2.T @ D2)
    return spsolve(A.tocsc(), weight * y)


def _fit_pspline(
    mz: np.ndarray, ppm: np.ndarray, weight: np.ndarray,
    bin_width_da: float, lam1: float, lam2: float, degree: int = 3,
) -> Callable[[np.ndarray], np.ndarray]:
    """Penalized B-spline (P-spline, Eilers & Marx 1996) regression: fit B-spline
    basis coefficients by penalized weighted least squares, penalizing first- and
    second-order differences between *adjacent coefficients* -- not raw data, unlike
    `_derivative_penalized_smooth` -- so the result stays a genuine smooth spline
    (continuous up to the `degree`-1 derivative) throughout, instead of a penalized
    set of node values connected by straight lines. Interior knots are placed evenly
    every `bin_width_da` -- dense, per the P-spline recipe, since the penalty (not
    knot placement) is what does the smoothing, avoiding the sparse/noisy-node
    oscillation a `natural_cubic_spline` (an exact interpolant, no penalty at all)
    can show.

    Boundary knots are clamped (`degree + 1` repeats), and the first two and last two
    basis coefficients are tied together -- for a clamped B-spline, `f'(boundary)` is
    exactly proportional to the difference between its two boundary coefficients, so
    tying them forces a flat (zero first-derivative) approach at both ends without
    pinning the curve to any particular boundary *value*.
    """
    lo, hi = float(mz.min()), float(mz.max())
    n_interior = max(1, int(np.ceil((hi - lo) / bin_width_da)) - 1)
    interior_knots = np.linspace(lo, hi, n_interior + 2)[1:-1]
    knots = np.concatenate([[lo] * (degree + 1), interior_knots, [hi] * (degree + 1)])
    n_basis = len(knots) - degree - 1

    B = BSpline.design_matrix(mz, knots, degree)  # sparse (n_samples, n_basis)

    # Tie coefficient 0 to 1, and n_basis-2 to n_basis-1, via a (n_basis, n_reduced)
    # 0/1 selection matrix -- R @ c_reduced expands back to the full (tied) `c`.
    n_reduced = n_basis - 2
    cols = np.concatenate([[0], np.arange(n_reduced), [n_reduced - 1]])
    R = csr_matrix((np.ones(n_basis), (np.arange(n_basis), cols)), shape=(n_basis, n_reduced))

    BR = B @ R
    W = diags(weight)
    D1 = diags([-np.ones(n_reduced - 1), np.ones(n_reduced - 1)], offsets=[0, 1], shape=(n_reduced - 1, n_reduced))
    penalty = lam1 * (D1.T @ D1)
    if lam2 and n_reduced >= 3:
        D2 = diags(
            [np.ones(n_reduced - 2), -2 * np.ones(n_reduced - 2), np.ones(n_reduced - 2)],
            offsets=[0, 1, 2], shape=(n_reduced - 2, n_reduced),
        )
        penalty = penalty + lam2 * (D2.T @ D2)

    A = (BR.T @ W @ BR) + penalty
    b = BR.T @ (weight * ppm)
    c_reduced = spsolve(A.tocsc(), b)
    c = R @ c_reduced

    spline = BSpline(knots, c, degree, extrapolate=False)
    return lambda x: spline(np.clip(x, lo, hi))


def fit_correction(
    df: pd.DataFrame, config: dict, weights: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ppm-error-vs-m/z correction function.

    `config["model"]` selects between `"global_median"` (a single constant),
    `"binned_median"` (m/z-binned median, linearly interpolated between bin centers,
    constant beyond the outermost bins), `"pspline_derivative_penalized"` (see
    below), `"natural_cubic_spline"` (smoother alternative to `binned_median`'s
    jagged `np.interp` line, but exact-interpolating and prone to Runge's-phenomenon
    -style oscillation between sparse/noisy nodes -- see
    `"pspline_derivative_penalized"` for a non-oscillating alternative), and
    `"xgboost_derivative_penalized"` (see below) -- a plain if/elif on the string, no
    dynamic import. Every occupied bin is used as a node regardless of how many PSMs
    fall in it -- no minimum-count threshold -- since `np.interp`/the spline already
    hold the value of the nearest node constant past the edges, so a
    sparsely-populated bin at the edge degrades gracefully instead of needing a
    separate fallback.

    `"pspline_derivative_penalized"` (`_fit_pspline`) fits a genuine penalized B-spline
    (P-spline) directly on the raw (not binned-median) data: dense interior knots
    every `bin_width_da`, `config["lam1"]`/`config["lam2"]` (both default `0.0`)
    penalizing first/second differences between adjacent B-spline coefficients, and
    the two boundary coefficient pairs tied together so the curve flattens (zero
    first derivative, not necessarily zero value) at both ends. Unlike
    `"natural_cubic_spline"` (an exact interpolant with no penalty at all) this
    doesn't oscillate between sparse/noisy nodes (e.g. a pooled precursor+fragment
    fit, where fragments extend well past the precursor m/z range into sparser
    territory), while staying a true smooth spline throughout -- unlike
    `_derivative_penalized_smooth`-based approaches, which penalize discrete *node
    values* and connect them with straight lines (only C0 continuous, visible kinks
    at bin boundaries), this penalizes spline *coefficients*, so the result is C2
    continuous everywhere, no kinks.

    `"natural_cubic_spline"` bins the same way `binned_median` does (`bin_width_da`),
    then groups those bins 3-wide (one node per 3 `bin_width_da` bins) and takes the
    per-group median as a node point. A cubic spline through those nodes is fit with
    the first derivative clamped to zero at both ends (`bc_type=((1, 0), (1, 0))`),
    so it flattens smoothly (no kink, unlike a plain clip) into a constant beyond the
    outermost node -- evaluated by clipping the input m/z into `[node_mz.min(),
    node_mz.max()]` before calling the spline. Nothing constrains the curve *between*
    nodes, though, so uneven/sparse node spacing can make it swing wildly well inside
    the fitted range, not just at the clipped boundary.

    `"xgboost_derivative_penalized"` bins the same way `binned_median` does
    (`bin_width_da`), but each node's y-value is an `XGBRegressor` prediction at the
    bin center -- fit once on every row, not per-bin -- rather than a plain per-bin
    median, so the node values can capture nonlinear structure a median can't. Those
    predicted node values are then smoothed by `_derivative_penalized_smooth`
    (`config["lam"]`, weighted by bin occupancy) before `np.interp`, since an
    unconstrained per-bin xgboost prediction can be as jagged as `binned_median`'s
    raw median -- the derivative penalty is what keeps the fitted curve smooth
    despite xgboost's flexibility. `config["xgboost_kwargs"]` (optional dict)
    overrides `XGBRegressor`'s defaults (`n_estimators=200, max_depth=3,
    learning_rate=0.05, reg_lambda=1.0`).

    `weights` (optional, one entry per `df` row, e.g. a selection strategy's
    per-survivor neighborhood density) turns every per-node/global `ppm` median above
    into a weighted median instead (and feeds `XGBRegressor.fit`'s `sample_weight`
    for `xgboost_derivative_penalized`) -- node/bin *placement* (`bin_centers`/
    `node_mz`) stays an unweighted median of `precursor_mz`, only the fitted `ppm`
    value at each node is reweighted. Omitted (default `None`) reproduces today's
    unweighted behavior exactly, so existing callers (e.g. `recalibrate()`) are
    unaffected.
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

    if model == "pspline_derivative_penalized":
        mz = df["precursor_mz"].to_numpy()
        bin_width = config["bin_width_da"]
        lam1 = config.get("lam1", 0.0)
        lam2 = config.get("lam2", 0.0)
        degree = config.get("degree", 3)
        weight = np.ones_like(ppm) if weights is None else weights
        return _fit_pspline(mz, ppm, weight, bin_width, lam1, lam2, degree)

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

    if model == "xgboost_derivative_penalized":
        mz = df["precursor_mz"].to_numpy()
        bin_width = config["bin_width_da"]
        lam = config["lam"]
        xgb_kwargs = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "reg_lambda": 1.0,
            **config.get("xgboost_kwargs", {}),
        }

        regressor = XGBRegressor(**xgb_kwargs)
        regressor.fit(mz.reshape(-1, 1), ppm, sample_weight=weights)

        bin_idx = np.floor(mz / bin_width).astype(np.int64)
        counts = pd.Series(bin_idx).value_counts().sort_index()
        bin_centers = (counts.index.to_numpy() + 0.5) * bin_width
        node_ppm = regressor.predict(bin_centers.reshape(-1, 1))
        smoothed = _derivative_penalized_smooth(node_ppm, counts.to_numpy(dtype=np.float64), lam)
        return lambda mz: np.interp(mz, bin_centers, smoothed)

    raise ValueError(f"unknown recalibration model: {model!r}")


if False:
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

def _plot_recalibration_fit(
    plot_path: str | Path,
    precursor_mz: np.ndarray,
    precursor_ppm: np.ndarray,
    fragment_mz: np.ndarray,
    fragment_ppm: np.ndarray,
    correction: Callable[[np.ndarray], np.ndarray],
    config: dict,
) -> None:
    """Scatter both precursor and fragment ppm-error clouds (still the 1D case --
    every current model regresses ppm against m/z alone; a model using more
    dimensions would need a scatterplot-matrix instead, not built since nothing uses
    more than one dimension yet) plus the one fitted trendline shared by both
    (see `recalibrate()`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        fragment_mz, fragment_ppm,
        s=2, alpha=0.15, color="#E69F00", linewidths=0,
        label=f"fragments (n={len(fragment_mz):_})",
    )
    ax.scatter(
        precursor_mz, precursor_ppm,
        s=4, alpha=0.35, color="#0072B2", linewidths=0,
        label=f"precursors (n={len(precursor_mz):_})",
    )

    line_mz = np.linspace(
        min(precursor_mz.min(), fragment_mz.min()),
        max(precursor_mz.max(), fragment_mz.max()),
        400,
    )
    ax.plot(line_mz, correction(line_mz), color="black", linewidth=2.5, label="fitted correction")
    ax.axhline(0, color="#808080", linewidth=1, linestyle="--")
    ax.set_xlabel("m/z")
    ax.set_ylabel("ppm error")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(
        f"Recalibration fit -- model={config['model']}, "
        f"tolerance_percentiles={config['tolerance_percentiles']}"
    )
    fig.tight_layout()

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def recalibrate(
    sage_results_tsv: str | Path,
    matched_fragments: str | Path,
    tof2mz: np.ndarray,
    config: dict,
    fdr: float,
    plot_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the ppm correction from confident PSMs and apply it to the tof2mz lookup
    array (fragments only -- see searchops_pandas_dictodot_multigit.md / sage_rescoring.md
    for why precursor mz needs its own separate correction step, done elsewhere by
    `recalibrate-precursor-mz`).

    One `fit_correction` model is fit directly on precursor and fragment (mz, ppm)
    data pooled together, with no per-type adjustment at all -- fragment residuals
    via `matched_fragments.sage.tsv`'s `fragment_mz_experimental`/
    `fragment_mz_calculated`, since Sage's own per-PSM `fragment_ppm` is an
    intensity-weighted mean of *absolute* error and cannot give a signed residual.
    Pooling (rather than fitting on precursor m/z alone and evaluating on fragments'
    much wider range, as an earlier version of this function did) means the fit's
    domain spans the *union* of precursor and fragment m/z, so it doesn't silently
    extrapolate past its own training range. `precursor_tol`/`fragment_tol` are both
    `config["tolerance_percentiles"]` applied to each type's own residual under this
    one shared correction, so the two windows can still differ (fragments may simply
    be noisier around the same fitted trend) even though nothing about the fit itself
    is type-specific.

    `plot_path`, if given, saves a diagnostic scatter+trendline plot
    (`_plot_recalibration_fit`) from this exact fit -- no re-reading or re-fitting.
    """
    df = filter_top_psms(sage_results_tsv, fdr)
    fragments = _confident_matched_fragments(matched_fragments, df["psm_id"])

    pooled_df = pd.DataFrame({
        "precursor_mz": np.concatenate([
            df["precursor_mz"].to_numpy(),
            fragments["fragment_mz_experimental"].to_numpy(),
        ]),
        "precursor_ppm": np.concatenate([
            df["precursor_ppm"].to_numpy(),
            fragments["fragment_ppm"].to_numpy(),
        ]),
    })
    correction = fit_correction(pooled_df, config)

    residual_precursor_ppm = df["precursor_ppm"].to_numpy() - correction(
        df["precursor_mz"].to_numpy()
    )
    residual_fragment_ppm = fragments["fragment_ppm"].to_numpy() - correction(
        fragments["fragment_mz_experimental"].to_numpy()
    )
    lo_pct, hi_pct = config["tolerance_percentiles"]
    precursor_tol = [
        float(np.percentile(residual_precursor_ppm, lo_pct)),
        float(np.percentile(residual_precursor_ppm, hi_pct)),
    ]
    fragment_tol = [
        float(np.percentile(residual_fragment_ppm, lo_pct)),
        float(np.percentile(residual_fragment_ppm, hi_pct)),
    ]

    new_tof2mz = tof2mz / (1.0 + correction(tof2mz) * 1e-6)

    if plot_path is not None:
        _plot_recalibration_fit(
            plot_path,
            df["precursor_mz"].to_numpy(), df["precursor_ppm"].to_numpy(),
            fragments["fragment_mz_experimental"].to_numpy(), fragments["fragment_ppm"].to_numpy(),
            correction,
            config,
        )

    tolerance = {
        "precursor_tol": {"ppm": precursor_tol},
        "fragment_tol": {"ppm": fragment_tol},
    }
    return new_tof2mz.astype(np.float32), tolerance
