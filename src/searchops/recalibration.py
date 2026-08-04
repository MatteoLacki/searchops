"""Fit and apply a precursor m/z recalibration from confident SAGE PSMs.

Sage's `precursor_ppm` (`feature.delta_mass` in sage-core's `scoring.rs`:
`(expmass - calcmass - isotope_error) * 2e6 / (expmass - isotope_error + calcmass)`)
is positive when the observed mass is heavier than theoretical -- so correcting a
tof2mz-derived m/z means *dividing* by `(1 + ppm/1e6)`, not adding it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numba
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, CubicSpline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from xgboost import DMatrix, XGBRegressor

from pandas_ops.io import read_df

PROTON_MASS = 1.00727646688


def filter_top_psms(sage_results_tsv: str | Path, fdr: float) -> pd.DataFrame:
    """Top-ranked, FDR-confident, target-only PSMs, with a `precursor_mz` column added.

    `peptide_q <= fdr` alone lets a few decoys through near the threshold (target-decoy
    competition doesn't guarantee every sub-threshold row is a target), so `label == 1`
    is checked explicitly too.
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
    column added. Sage's per-PSM `fragment_ppm` (in `results.sage.tsv`) is an
    intensity-weighted mean of *absolute* error and can't be used as a residual --
    `matched_fragments.sage.tsv`'s per-fragment `fragment_mz_calculated`/
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
    """Weighted Whittaker/P-spline smoother: minimizes `sum(w_i (f_i-y_i)^2) + lam1 *
    sum((f_{i+1}-f_i)^2) + lam2 * sum((f_{i+1}-2f_i+f_{i-1})^2)` over `f`, via a
    banded solve. `lam1` penalizes slope changes between neighbors (first
    derivative); `lam2` penalizes curvature (second derivative) -- needed because a
    first-derivative-only penalty still lets segments swing independently between
    sparse/noisy nodes, as long as no single slope is individually large.
    `lam1=lam2=0` reproduces `y` exactly. Assumes `y`/`weight` are already ordered
    along the penalty axis (e.g. node index by increasing m/z).
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
    """Penalized B-spline (P-spline, Eilers & Marx 1996): fit B-spline coefficients
    by penalized weighted least squares, penalizing first/second differences between
    *adjacent coefficients* (not raw data, unlike `_derivative_penalized_smooth`) --
    so the result is a genuine C(`degree`-1)-continuous spline, not a penalized set
    of node values joined by straight lines. Interior knots are dense (every
    `bin_width_da`), since the penalty rather than knot placement does the
    smoothing, avoiding the sparse/noisy-node oscillation an unpenalized exact
    interpolant (`natural_cubic_spline`) can show.

    Boundary knots are clamped, and the first two / last two basis coefficients are
    each tied together -- for a clamped B-spline, `f'(boundary)` is proportional to
    the difference between its two boundary coefficients, so tying them forces a
    flat (zero first-derivative) approach at both ends without pinning the curve to
    a specific boundary value.
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

    `config["model"]` (plain if/elif, no dynamic import) selects between:
    - `"global_median"`: a single constant.
    - `"binned_median"`: m/z-binned median, `np.interp`-ed between bin centers,
      constant beyond the outermost bins.
    - `"pspline_derivative_penalized"`: see below.
    - `"natural_cubic_spline"`: smoother than `binned_median`'s jagged line, but an
      exact interpolant prone to Runge's-phenomenon oscillation between sparse/noisy
      nodes -- see `"pspline_derivative_penalized"` for a non-oscillating alternative.
    - `"xgboost_derivative_penalized"`: see below.

    Every occupied bin is used as a node regardless of count -- no minimum-count
    threshold -- since `np.interp`/the spline already hold the nearest node's value
    constant past the edges, so a sparse edge bin degrades gracefully.

    `"pspline_derivative_penalized"` (`_fit_pspline`) fits a genuine penalized
    B-spline directly on the raw (not binned-median) data: dense interior knots
    every `bin_width_da`, `config["lam1"]`/`config["lam2"]` (default `0.0`)
    penalizing first/second differences between adjacent coefficients, boundary
    coefficient pairs tied so the curve flattens (zero first derivative) at both
    ends. Unlike `"natural_cubic_spline"` this doesn't oscillate between
    sparse/noisy nodes (e.g. a pooled precursor+fragment fit, where fragments extend
    well past the precursor m/z range) while staying C2-continuous throughout --
    unlike `_derivative_penalized_smooth`-based approaches, which penalize discrete
    node values joined by straight lines (only C0, visible kinks), this penalizes
    spline coefficients directly.

    `"natural_cubic_spline"` bins like `binned_median` (`bin_width_da`), groups bins
    3-wide, takes each group's median as a node, then fits a cubic spline through
    those nodes with the first derivative clamped to zero at both ends (flattens
    smoothly beyond the outermost node, evaluated by clipping input m/z into the
    node range first). Nothing constrains the curve *between* nodes, so
    uneven/sparse spacing can make it swing wildly well inside the fitted range.

    `"xgboost_derivative_penalized"` bins like `binned_median`, but each node's
    y-value is an `XGBRegressor` prediction at the bin center (fit once on every
    row, not per-bin) rather than a plain median, so nodes can capture nonlinear
    structure a median can't. Those predicted nodes are then smoothed by
    `_derivative_penalized_smooth` (`config["lam"]`, weighted by bin occupancy)
    before `np.interp` -- the derivative penalty is what keeps the curve smooth
    despite xgboost's flexibility. `config["xgboost_kwargs"]` overrides
    `XGBRegressor`'s defaults (`n_estimators=200, max_depth=3, learning_rate=0.05,
    reg_lambda=1.0`).

    `weights` (optional, one entry per `df` row) turns every median above into a
    weighted median (and feeds `XGBRegressor.fit`'s `sample_weight`) -- node/bin
    *placement* stays an unweighted median of `precursor_mz`, only the fitted `ppm`
    value is reweighted. `None` (default) reproduces the unweighted behavior exactly.
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


def fit_additive_correction(
    df: pd.DataFrame,
    dims: list[str],
    target: str,
    config: dict,
    weights: np.ndarray | None = None,
) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], float]:
    """Multi-dimensional generalization of `fit_correction()`: a GAM-style additive
    model `target ~= f_1(dims[0]) + f_2(dims[1]) + ...` (e.g. ppm error as a function
    of m/z, ion mobility, and retention time at once) instead of `fit_correction`'s
    m/z-only fit. Wholly new, additive-only function -- `fit_correction()`'s contract
    and call sites are untouched.

    Returns `(components, bias)`: `components` has one plain, non-numba
    `Callable[[np.ndarray], np.ndarray]` per entry of `dims` (same contract as
    `_fit_pspline`'s return value), not a single combined callable; `bias` is the
    constant term common to the whole fit, kept as a plain `float` rather than
    folded into one component, since it isn't a function of anything. Reconstruct
    via `bias + sum(components[d](x_d) for d, x_d in zip(dims, columns))`. Each
    component drops unmodified into `to_numba_correction(component, dim_lo, dim_hi,
    n_grid)` (summing/numba-wiring across dims is separate, later work).

    `config["model"]`:

    `"xgboost_additive"`: a single `XGBRegressor` with `max_depth` forced to `1`
    across all `dims` -- depth-1 trees split on exactly one feature each, so the
    ensemble sum is an *exact* additive decomposition (zero cross-terms) via
    TreeSHAP (`booster.predict(DMatrix(...), pred_contribs=True)`; spot-checked to
    ~1e-6 reconstruction error). Each component is a closure that builds a synthetic
    grid (that dimension varying, every other held at its training-data median --
    irrelevant to a depth-1 split on this dimension, so the placeholder only affects
    robustness, not correctness) and reads off that dimension's SHAP column. Bias is
    returned unfolded. `config["xgboost_kwargs"]` overrides the defaults
    (`n_estimators=1200, learning_rate=0.03, subsample=0.8, reg_lambda=2.0` -- depth-1
    trees need far more of them than `xgboost_derivative_penalized`'s depth-3
    default); an explicit `max_depth != 1` raises `ValueError` rather than silently
    breaking the additivity guarantee. No internal train/valid split or early
    stopping -- a single fixed-round fit, fine for test-only/not-yet-pipeline-wired.

    `"pspline_additive"`: classical Gauss-Seidel backfitting using `_fit_pspline` as
    the per-dimension smoother, `config.get("backfit_iters", 15)` passes over `dims`,
    one shared `bin_width_da`/`lam1`/`lam2`/`degree` across all dimensions (no
    per-dim hyperparameters yet). `intercept` is fixed up front as the weighted mean
    of `target`; each round, each dimension refits against the partial residual
    (`target` minus intercept minus every *other* dimension's current fit) and
    re-centers to weighted-mean zero -- additive models are identifiable only up to
    constants that shift between components while cancelling in the sum, so without
    re-centering the components would drift instead of converging. `intercept` is
    returned as `bias`. Fixed iteration count, no convergence check -- a
    max-abs-change early exit would be a trivial future addition.

    `weights` (optional, one entry per `df` row) mirrors `fit_correction`'s own:
    `None` means unweighted; otherwise it feeds `XGBRegressor.fit`'s `sample_weight`
    or every weighted-mean/`_fit_pspline` call in the pspline branch.
    """
    if not dims:
        raise ValueError("dims must be non-empty")
    model = config["model"]
    y = df[target].to_numpy(dtype=np.float64)
    weight = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)

    if model == "xgboost_additive":
        user_kwargs = config.get("xgboost_kwargs", {})
        if "max_depth" in user_kwargs and user_kwargs["max_depth"] != 1:
            raise ValueError(
                "xgboost_additive requires max_depth=1 for the additive-decomposition "
                f"guarantee to hold; got xgboost_kwargs['max_depth']={user_kwargs['max_depth']!r}"
            )
        xgb_kwargs = {
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "reg_lambda": 2.0,
            **user_kwargs,
            "max_depth": 1,  # forced: see docstring
        }

        # A DataFrame (not a raw ndarray) so real column names attach to the
        # booster -- needed so the later `DMatrix(grid_df)` predict-contribs calls
        # don't hit a feature-names mismatch against anonymous f0/f1/... names.
        feature_df = df[dims].astype(np.float64)
        regressor = XGBRegressor(**xgb_kwargs)
        regressor.fit(feature_df, y, sample_weight=weights)
        booster = regressor.get_booster()

        medians = {d: float(df[d].median()) for d in dims}
        train_contribs = booster.predict(DMatrix(feature_df), pred_contribs=True)
        bias = float(np.mean(train_contribs[:, -1]))  # row-invariant base-score term

        def _make_component(dim_index: int, dim_name: str) -> Callable[[np.ndarray], np.ndarray]:
            def component(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                grid = np.empty((x.shape[0], len(dims)), dtype=np.float64)
                for k, other in enumerate(dims):
                    grid[:, k] = x if other == dim_name else medians[other]
                grid_df = pd.DataFrame(grid, columns=dims)
                contribs = booster.predict(DMatrix(grid_df), pred_contribs=True)
                return contribs[:, dim_index]
            return component

        components = {d: _make_component(j, d) for j, d in enumerate(dims)}
        return components, bias

    if model == "pspline_additive":
        bin_width_da = config["bin_width_da"]
        lam1 = config.get("lam1", 0.0)
        lam2 = config.get("lam2", 0.0)
        degree = config.get("degree", 3)
        n_iters = config.get("backfit_iters", 15)
        if n_iters < 1:
            raise ValueError(f"backfit_iters must be >= 1, got {n_iters!r}")

        columns = {d: df[d].to_numpy(dtype=np.float64) for d in dims}
        intercept = float(np.average(y, weights=weight))
        fitted_values = {d: np.zeros(len(y), dtype=np.float64) for d in dims}
        components: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

        for _ in range(n_iters):
            for d in dims:
                others_sum = np.zeros(len(y), dtype=np.float64)
                for other in dims:
                    if other != d:
                        others_sum += fitted_values[other]
                partial_residual = y - intercept - others_sum
                raw_component = _fit_pspline(
                    columns[d], partial_residual, weight, bin_width_da, lam1, lam2, degree
                )
                raw_values = raw_component(columns[d])
                mean_d = float(np.average(raw_values, weights=weight))
                fitted_values[d] = raw_values - mean_d
                # f/delta default args (not a closure over the loop variables by
                # reference) so each dimension's lambda keeps its own round's
                # values, not whatever raw_component/mean_d end up as after the
                # loop finishes (the classic late-binding closure bug).
                components[d] = lambda x, f=raw_component, delta=mean_d: f(x) - delta

        return components, intercept

    raise ValueError(f"unknown additive recalibration model: {model!r}")


def _plot_recalibration_fit(
    plot_path: str | Path,
    precursor_mz: np.ndarray,
    precursor_ppm: np.ndarray,
    fragment_mz: np.ndarray,
    fragment_ppm: np.ndarray,
    precursor_correction: Callable[[np.ndarray], np.ndarray],
    fragment_correction: Callable[[np.ndarray], np.ndarray],
    config: dict,
) -> None:
    """Scatter both precursor and fragment ppm-error clouds plus each type's own
    fitted trendline (still 1D -- every current model regresses ppm against m/z
    alone). When `fit_separately` is off, `precursor_correction`/`fragment_correction`
    are the same object (see `recalibrate()`) so the two lines coincide."""
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

    fragment_line_mz = np.linspace(fragment_mz.min(), fragment_mz.max(), 400)
    ax.plot(
        fragment_line_mz, fragment_correction(fragment_line_mz),
        color="black", linewidth=2.5, linestyle="--" if fragment_correction is not precursor_correction else "-",
        label="fitted correction (fragments)" if fragment_correction is not precursor_correction else "fitted correction",
    )
    if fragment_correction is not precursor_correction:
        precursor_line_mz = np.linspace(precursor_mz.min(), precursor_mz.max(), 400)
        ax.plot(
            precursor_line_mz, precursor_correction(precursor_line_mz),
            color="black", linewidth=2.5, linestyle="-",
            label="fitted correction (precursors)",
        )
    ax.axhline(0, color="#808080", linewidth=1, linestyle=":")
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


def to_numba_correction(
    correction: Callable[[np.ndarray], np.ndarray],
    mz_lo: float, mz_hi: float, n_grid: int = 2000,
) -> Callable[[float], float]:
    """Distill any `fit_correction()`-style closure into a numba `@njit` scalar
    function, callable directly from other numba-compiled (including
    `parallel=True`/`prange`) code -- none of `CubicSpline`/`BSpline`/`XGBRegressor`
    are numba-callable themselves, but every model reduces to a plain
    array-in-array-out function, so sampling it on a dense grid into a lookup table
    works uniformly regardless of which model produced it.

    Deliberately *not* `np.interp` (an O(log n) binary search per call):
    `breakpoints` is evenly spaced by construction, so the containing interval is a
    single O(1) division -- ~35-40x faster in practice, benchmarked on this exact
    use case.

    `breakpoints`/`values` are baked in via closure so the call site is just
    `corrector(mz)`, at the cost of a fresh one-time JIT compile per call (no
    `cache=True`: a closure over array freevars can't be meaningfully cached across
    process restarts) -- the normal numba usage pattern regardless (compile once,
    apply to many values after).

    `mz_lo`/`mz_hi` should cover the full range this will ever be queried at, not
    just the range it was fit on -- e.g. in `recalibrate()`, the union of the
    fitting data's own m/z range and the full `tof2mz` lookup table's range, since
    the latter can extend further. Whatever `correction` does beyond its own fitted
    range (hold constant, taper flat, etc.) carries through automatically, since
    `values` comes from calling `correction` on the grid, not reimplemented here.
    """
    breakpoints = np.linspace(mz_lo, mz_hi, n_grid)
    values = np.asarray(correction(breakpoints), dtype=np.float64)
    spacing = (mz_hi - mz_lo) / (n_grid - 1)
    inv_spacing = 1.0 / spacing
    n = n_grid

    @numba.njit(nogil=True)
    def corrector(mz):
        idx_f = (mz - mz_lo) * inv_spacing
        if idx_f <= 0.0:
            return values[0]
        if idx_f >= n - 1:
            return values[n - 1]
        idx = int(idx_f)
        frac = idx_f - idx
        return values[idx] * (1.0 - frac) + values[idx + 1] * frac

    return corrector


@numba.njit(parallel=True, nogil=True)
def _apply_numba_correction(mz_array: np.ndarray, corrector) -> np.ndarray:
    """Apply a `to_numba_correction()` scalar corrector to a whole array, in
    parallel -- used wherever `recalibrate()` needs array-in-array-out behavior
    (residuals, `tof2mz`, the plot); the scalar `corrector` itself is what's meant
    for embedding directly in other multithreaded numba code.
    """
    out = np.empty(mz_array.shape, dtype=np.float64)
    for i in numba.prange(mz_array.shape[0]):
        out[i] = corrector(mz_array[i])
    return out


def recalibrate(
    sage_results_tsv: str | Path,
    matched_fragments: str | Path,
    tof2mz: np.ndarray,
    config: dict,
    fdr: float,
    plot_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the ppm correction from confident PSMs and apply it to the tof2mz lookup
    array (fragments only -- precursor mz gets its own separate correction step,
    done elsewhere by `recalibrate-precursor-mz`).

    By default one `fit_correction` model is fit on precursor and fragment (mz,
    ppm) data pooled together -- fragment residuals via `matched_fragments.sage.tsv`'s
    `fragment_mz_experimental`/`fragment_mz_calculated`, since Sage's own per-PSM
    `fragment_ppm` is an intensity-weighted mean of *absolute* error and can't give
    a signed residual. Pooling means the fit's domain spans the *union* of
    precursor and fragment m/z, so it doesn't silently extrapolate past its own
    training range.

    `config["fit_separately"]` (default `False`) fits the same model/hyperparameters
    independently on precursor-only and fragment-only data instead -- two correctors,
    each responsible only for its own residuals; `tof2mz` always uses the fragment
    corrector. `precursor_tol`/`fragment_tol` are both `config["tolerance_percentiles"]`
    applied to each type's own residual, so the two windows can differ even pooled.

    `plot_path`, if given, saves a diagnostic plot (`_plot_recalibration_fit`) from
    this exact fit -- no re-reading or re-fitting.

    Every corrector is immediately distilled via `to_numba_correction` and used as
    that from here on -- one numba code path by default, so whatever this fits is
    already in a form other numba-compiled pipeline code can embed directly.
    """
    df = filter_top_psms(sage_results_tsv, fdr)
    fragments = _confident_matched_fragments(matched_fragments, df["psm_id"])

    precursor_mz = df["precursor_mz"].to_numpy()
    precursor_ppm = df["precursor_ppm"].to_numpy()
    fragment_mz = fragments["fragment_mz_experimental"].to_numpy()
    fragment_ppm = fragments["fragment_ppm"].to_numpy()

    n_grid = config.get("numba_grid_points", 2000)

    def build_correction(fitted, mz_lo, mz_hi):
        numba_correction = to_numba_correction(fitted, float(mz_lo), float(mz_hi), n_grid)
        return lambda mz: _apply_numba_correction(np.asarray(mz, dtype=np.float64), numba_correction)

    if config.get("fit_separately", False):
        precursor_fit = fit_correction(
            pd.DataFrame({"precursor_mz": precursor_mz, "precursor_ppm": precursor_ppm}), config
        )
        fragment_fit = fit_correction(
            pd.DataFrame({"precursor_mz": fragment_mz, "precursor_ppm": fragment_ppm}), config
        )
        precursor_correction = build_correction(precursor_fit, precursor_mz.min(), precursor_mz.max())
        fragment_correction = build_correction(
            fragment_fit, min(fragment_mz.min(), np.min(tof2mz)), max(fragment_mz.max(), np.max(tof2mz))
        )
    else:
        pooled_df = pd.DataFrame({
            "precursor_mz": np.concatenate([precursor_mz, fragment_mz]),
            "precursor_ppm": np.concatenate([precursor_ppm, fragment_ppm]),
        })
        fitted_correction = fit_correction(pooled_df, config)
        shared_correction = build_correction(
            fitted_correction,
            min(pooled_df["precursor_mz"].min(), np.min(tof2mz)),
            max(pooled_df["precursor_mz"].max(), np.max(tof2mz)),
        )
        precursor_correction = shared_correction
        fragment_correction = shared_correction

    residual_precursor_ppm = precursor_ppm - precursor_correction(precursor_mz)
    residual_fragment_ppm = fragment_ppm - fragment_correction(fragment_mz)
    lo_pct, hi_pct = config["tolerance_percentiles"]
    precursor_tol = [
        float(np.percentile(residual_precursor_ppm, lo_pct)),
        float(np.percentile(residual_precursor_ppm, hi_pct)),
    ]
    fragment_tol = [
        float(np.percentile(residual_fragment_ppm, lo_pct)),
        float(np.percentile(residual_fragment_ppm, hi_pct)),
    ]

    new_tof2mz = tof2mz / (1.0 + fragment_correction(tof2mz) * 1e-6)

    if plot_path is not None:
        _plot_recalibration_fit(
            plot_path,
            precursor_mz, precursor_ppm,
            fragment_mz, fragment_ppm,
            precursor_correction, fragment_correction,
            config,
        )

    tolerance = {
        "precursor_tol": {"ppm": precursor_tol},
        "fragment_tol": {"ppm": fragment_tol},
    }
    return new_tof2mz.astype(np.float32), tolerance


def _hist_panel(ax, before, after, lo_tol, hi_tol, xlabel, title) -> None:
    ax.hist(
        before, bins=200, density=True, alpha=0.5, color="#E69F00",
        label=f"before recalibration (n={len(before):_})",
    )
    ax.hist(
        after, bins=200, density=True, alpha=0.5, color="#0072B2",
        label=f"after recalibration (n={len(after):_})",
    )
    ax.axvline(0, color="#808080", linewidth=1, linestyle=":")
    ax.axvline(lo_tol, color="#D55E00", linewidth=1.5, linestyle="--", label="tolerance")
    ax.axvline(hi_tol, color="#D55E00", linewidth=1.5, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=9, loc="best")
    ax.set_title(title)


def plot_recalibrated_ppm(
    initial_sage_results_tsv: str | Path,
    sage_results_tsv: str | Path,
    initial_matched_fragments: str | Path,
    matched_fragments: str | Path,
    tolerance: dict,
    fdr: float,
    plot_path: str | Path,
) -> None:
    """Two-panel plot: marginal `precursor_ppm` (top) and `fragment_ppm` (bottom)
    distributions, both unconditional on m/z (plain 1D histograms, unlike
    `_plot_recalibration_fit`'s scatter-vs-mz), overlaid before vs. after
    recalibration.

    `initial_sage_results_tsv`/`initial_matched_fragments` are the *first* SAGE
    pass's outputs -- the uncorrected search on `recalibration_precursor_selection`'s
    subset, the same data `recalibrate()` fits from. `sage_results_tsv`/
    `matched_fragments` are the *second*, final pass's outputs -- full precursor
    population, already searched with the corrected tof2mz/tolerances baked in.
    These don't share a precursor population (subset vs. full), so each panel is a
    shape/spread comparison, not a paired one.

    `tolerance["precursor_tol"]["ppm"]`/`["fragment_tol"]["ppm"]` (the dict
    `recalibrate()` returned, that the second pass actually ran with) are drawn as
    reference bands.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial_df = filter_top_psms(initial_sage_results_tsv, fdr)
    final_df = filter_top_psms(sage_results_tsv, fdr)
    initial_precursor_ppm = initial_df["precursor_ppm"].to_numpy()
    final_precursor_ppm = final_df["precursor_ppm"].to_numpy()

    initial_fragment_ppm = _confident_matched_fragments(
        initial_matched_fragments, initial_df["psm_id"]
    )["fragment_ppm"].to_numpy()
    final_fragment_ppm = _confident_matched_fragments(
        matched_fragments, final_df["psm_id"]
    )["fragment_ppm"].to_numpy()

    precursor_lo, precursor_hi = tolerance["precursor_tol"]["ppm"]
    fragment_lo, fragment_hi = tolerance["fragment_tol"]["ppm"]

    fig, (ax_precursor, ax_fragment) = plt.subplots(2, 1, figsize=(10, 10))
    _hist_panel(
        ax_precursor, initial_precursor_ppm, final_precursor_ppm, precursor_lo, precursor_hi,
        "precursor ppm error", f"Precursor ppm error distribution, before vs after recalibration (fdr={fdr})",
    )
    _hist_panel(
        ax_fragment, initial_fragment_ppm, final_fragment_ppm, fragment_lo, fragment_hi,
        "fragment ppm error", f"Fragment ppm error distribution, before vs after recalibration (fdr={fdr})",
    )
    fig.tight_layout()

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
