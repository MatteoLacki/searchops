"""Fit and apply a precursor m/z recalibration from confident SAGE PSMs.

Sage's `precursor_ppm` (`feature.delta_mass` in sage-core's `scoring.rs`:
`(expmass - calcmass - isotope_error) * 2e6 / (expmass - isotope_error + calcmass)`)
is positive when the observed mass is heavier than theoretical -- so correcting a
tof2mz-derived m/z means *dividing* by `(1 + ppm/1e6)`, not adding it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import mmappet
import numpy as np
import pandas as pd
from numba_progress import ProgressBar
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from xgboost import DMatrix, XGBRegressor

from pandas_ops.io import read_df
from timstofu.mzrecalibration import EvenlySpacedLinearSpline, MzRecalibration

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
    D1 = diags([-np.ones(n - 1), np.ones(n - 1)], offsets=[0, 1], shape=(n - 1, n))  # type: ignore[reportArgumentType]
    A = diags(weight) + lam1 * (D1.T @ D1)
    if lam2 and n >= 3:
        D2 = diags(
            [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
            offsets=[0, 1, 2], shape=(n - 2, n),  # type: ignore[reportArgumentType]
        )
        A = A + lam2 * (D2.T @ D2)
    return np.asarray(spsolve(A.tocsc(), weight * y))


def _fit_pspline(
    mz: np.ndarray, ppm: np.ndarray, weight: np.ndarray,
    bin_width_da: float, lam1: float, lam2: float, degree: int = 3,
) -> tuple[BSpline, float, float]:
    """Penalized B-spline (P-spline, Eilers & Marx 1996): dense interior knots
    (every `bin_width_da`), boundary coefficient pairs tied so the curve flattens
    (zero first derivative) at both ends. Returns `(spline, lo, hi)`; evaluate as
    `spline(np.clip(x, lo, hi))` -- `extrapolate=False` NaNs outside `[lo, hi]`.
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
    D1 = diags([-np.ones(n_reduced - 1), np.ones(n_reduced - 1)], offsets=[0, 1], shape=(n_reduced - 1, n_reduced))  # type: ignore[reportArgumentType]
    penalty = lam1 * (D1.T @ D1)
    if lam2 and n_reduced >= 3:
        D2 = diags(
            [np.ones(n_reduced - 2), -2 * np.ones(n_reduced - 2), np.ones(n_reduced - 2)],
            offsets=[0, 1, 2], shape=(n_reduced - 2, n_reduced),  # type: ignore[reportArgumentType]
        )
        penalty = penalty + lam2 * (D2.T @ D2)

    A = (BR.T @ W @ BR) + penalty
    b = BR.T @ (weight * ppm)
    c_reduced = spsolve(A.tocsc(), b)
    c = R @ c_reduced

    spline = BSpline(knots, c, degree, extrapolate=False)
    return spline, lo, hi


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

        medians = {d: float(np.median(df[d].to_numpy())) for d in dims}
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
                spline, lo, hi = _fit_pspline(
                    columns[d], partial_residual, weight, bin_width_da, lam1, lam2, degree
                )
                raw_values = spline(np.clip(columns[d], lo, hi))
                mean_d = float(np.average(raw_values, weights=weight))
                fitted_values[d] = raw_values - mean_d
                # f/delta default args (not a closure over the loop variables by
                # reference) so each dimension's lambda keeps its own round's
                # values, not whatever spline/lo/hi/mean_d end up as after the
                # loop finishes (the classic late-binding closure bug).
                components[d] = (
                    lambda x, f=spline, l=lo, h=hi, delta=mean_d: f(np.clip(x, l, h)) - delta
                )

        return components, intercept

    raise ValueError(f"unknown additive recalibration model: {model!r}")


def _tolerance(residual: np.ndarray, percentiles: tuple[float, float]) -> dict:
    lo_pct, hi_pct = percentiles
    return {"ppm": [float(np.percentile(residual, lo_pct)), float(np.percentile(residual, hi_pct))]}


def _robust_sigma(residual: np.ndarray) -> float:
    """MAD-based robust scale estimate (`1.4826 * median(|x - median(x)|)`)
    -- mirrors `feature_prediction.tolerance.robust_sigma` (kept as a
    separate implementation, not a cross-package import, same reasoning as
    that module's own docstring for not sharing code with SAGE's Rust
    `LinearSpline`: different package, no shared dependency to hang it on).
    """
    residual = np.asarray(residual, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def _symmetric_tolerance(residual: np.ndarray, percentiles: tuple[float, float]) -> dict:
    """`median ± z * robust_sigma` (`z = norm.ppf(hi_pct / 100)`), symmetric
    by construction -- mirrors `feature_prediction.tolerance.
    symmetric_gaussian_tolerance`. Real F9477 precursor mass-error residuals
    are visibly right-skewed (2026-08-25 finding), largely an artifact of
    the fixed search window the calibration-pass anchors were drawn
    through, not something an empirical-percentile window should chase.
    """
    lo_pct, hi_pct = percentiles
    residual = np.asarray(residual, dtype=np.float64)
    center = float(np.median(residual))
    sigma = _robust_sigma(residual)
    z = float(norm.ppf(hi_pct / 100.0))
    return {"ppm": [center - z * sigma, center + z * sigma]}


def _select_tolerance(residual: np.ndarray, mz_config: dict) -> dict:
    """Dispatches on `mz_config["tolerance_method"]` (`"theoretic"`,
    default, or `"empiric"`) -- mirrors
    `feature_prediction.tolerance.select_tolerance`. `mz_config` is
    `config["mz"]` (`{"tolerance_percentiles": [...], "tolerance_method": ...}`).
    """
    percentiles = tuple(mz_config["tolerance_percentiles"])
    method = mz_config.get("tolerance_method", "theoretic")
    if method == "theoretic":
        return _symmetric_tolerance(residual, percentiles)
    if method == "empiric":
        return _tolerance(residual, percentiles)
    raise ValueError(f"unknown tolerance method {method!r}, expected 'theoretic' or 'empiric'")


def recalibrate_pmsms_mz(
    sage_results_tsv: str | Path,
    matched_fragments: str | Path,
    mz_pmsms: str | Path,
    config: dict,
    fdr: float,
    output_pmsms: str | Path,
    mz_recalibration_path: str | Path,
    plot_path: str | Path,
) -> dict:
    """Fit `config["fragment_model"]` on confident-PSM fragment residuals, grid-sample
    it into an `MzRecalibration` artifact, and apply it to `mz_pmsms`'s `mz` column in
    one pass, writing `output_pmsms` directly. The grid artifact is the sole
    serialization of the fit -- no separate raw-model dump for fragments.
    """
    from mmappet.fs import copy_dataset
    from timstofu.timstofmisc import apply_mz_recalibration

    from searchops.models import build_model

    df = filter_top_psms(sage_results_tsv, fdr)
    fragments = _confident_matched_fragments(matched_fragments, pd.Series(df["psm_id"]))
    fragment_mz = fragments["fragment_mz_experimental"].to_numpy()
    fragment_ppm = fragments["fragment_ppm"].to_numpy()

    model = build_model(config["fragment_model"]).fit(fragment_mz, fragment_ppm)

    mz_pmsms = Path(mz_pmsms)
    output_pmsms = Path(output_pmsms)
    if output_pmsms.exists():
        raise FileExistsError(f"{output_pmsms} already exists")
    schema = mmappet.str_to_schema((mz_pmsms / "schema.txt").read_text())
    columns = list(schema.columns)
    mz_idx = columns.index("mz")

    input_ds = mmappet.open_dataset_dct(mz_pmsms)
    raw_mz = input_ds["mz"]

    n_grid = config.get("numba_grid_points", 2000)
    mz_lo = min(float(fragment_mz.min()), float(np.min(raw_mz)))
    mz_hi = max(float(fragment_mz.max()), float(np.max(raw_mz)))
    breakpoints = np.linspace(mz_lo, mz_hi, n_grid)
    grid_dim = EvenlySpacedLinearSpline(mz_lo, mz_hi, model.predict(breakpoints))
    MzRecalibration(dims={"mz": grid_dim}).dump(mz_recalibration_path)
    corrector = grid_dim.njit_evaluator()

    fell_back = copy_dataset(mz_pmsms, output_pmsms, skip={f"{mz_idx}.bin"})
    if fell_back:
        print(
            f"warning: hard-link failed for one or more files under {mz_pmsms}; "
            "fell back to copy (uses extra disk space)",
            file=sys.stderr,
        )
    mz_path = output_pmsms / f"{mz_idx}.bin"
    with open(mz_path, "xb") as f:
        f.truncate(len(raw_mz) * raw_mz.dtype.itemsize)
    output_ds = mmappet.open_dataset_dct(output_pmsms, read_write=True)
    with ProgressBar(total=len(raw_mz), desc="recalibrate_pmsms_mz: applying correction") as progress:
        apply_mz_recalibration(raw_mz, corrector, 0.0, output_ds["mz"], progress)

    model.plot_fit(plot_path, fragment_mz, fragment_ppm, title="Fragment m/z recalibration fit")

    residual = fragment_ppm - model.predict(fragment_mz)
    return _select_tolerance(residual, config["mz"])


def recalibrate_precursors(
    sage_results_tsv: str | Path,
    precursors: str | Path,
    config: dict,
    fdr: float,
    output_precursors: str | Path,
    plot_path: str | Path,
    model_path: str | Path,
) -> dict:
    """Fit `config["precursor_model"]` on confident-PSM precursor residuals and
    apply it directly to `precursors`'s `mz` column, writing `output_precursors`.
    """
    from searchops.models import build_model

    df = filter_top_psms(sage_results_tsv, fdr)
    precursor_mz = df["precursor_mz"].to_numpy()
    precursor_ppm = df["precursor_ppm"].to_numpy()

    model = build_model(config["precursor_model"]).fit(precursor_mz, precursor_ppm)

    precursors_df = mmappet.open_dataset(Path(precursors))
    precursors_df = precursors_df.rename(columns={"mz": "mz_old"})
    old_mz = precursors_df["mz_old"].to_numpy()
    precursors_df["mz"] = old_mz / (1.0 + model.predict(old_mz) * 1e-6)
    with mmappet.DatasetWriter(Path(output_precursors), overwrite_dir=True) as writer:
        writer.append_df(precursors_df)

    model.plot_fit(plot_path, precursor_mz, precursor_ppm, title="Precursor m/z recalibration fit")
    model.save(model_path)

    residual = precursor_ppm - model.predict(precursor_mz)
    return _select_tolerance(residual, config["mz"])


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
    precursor_tolerance: dict,
    fragment_tolerance: dict,
    fdr: float,
    plot_path: str | Path,
) -> None:
    """Two-panel plot: marginal `precursor_ppm` (top) and `fragment_ppm` (bottom)
    distributions, both unconditional on m/z, overlaid before vs. after
    recalibration.

    `initial_sage_results_tsv`/`initial_matched_fragments` are the first SAGE
    pass's outputs (uncorrected search); `sage_results_tsv`/`matched_fragments`
    are the second, final pass's (full precursor population, corrected mz/
    tolerances baked in) -- different precursor populations, so each panel is a
    shape/spread comparison, not a paired one.

    `precursor_tolerance["ppm"]`/`fragment_tolerance["ppm"]` (from
    `recalibrate_precursors`/`recalibrate_pmsms_mz` respectively) are drawn as
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
        initial_matched_fragments, pd.Series(initial_df["psm_id"])
    )["fragment_ppm"].to_numpy()
    final_fragment_ppm = _confident_matched_fragments(
        matched_fragments, pd.Series(final_df["psm_id"])
    )["fragment_ppm"].to_numpy()

    precursor_lo, precursor_hi = precursor_tolerance["ppm"]
    fragment_lo, fragment_hi = fragment_tolerance["ppm"]

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
