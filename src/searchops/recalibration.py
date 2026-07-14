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

from pandas_ops.io import read_df

PROTON_MASS = 1.00727646688


def filter_top_psms(sage_results_tsv: str | Path, fdr: float) -> pd.DataFrame:
    """Top-ranked, FDR-confident PSMs, with a `precursor_mz` column added."""
    df = read_df(sage_results_tsv)
    df = df[(df["rank"] == 1) & (df["peptide_q"] <= fdr)].copy()
    df["precursor_mz"] = df["expmass"] / df["charge"] + PROTON_MASS
    return df


def fit_correction(df: pd.DataFrame, config: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ppm-error-vs-m/z correction function.

    `config["model"]` selects between `"global_median"` (a single constant) and
    `"binned_median"` (m/z-binned median, linearly interpolated between bin centers for
    smoothing) -- a plain if/elif on the string, no dynamic import.
    """
    model = config["model"]
    ppm = df["precursor_ppm"].to_numpy()

    if model == "global_median":
        offset = float(np.median(ppm))
        return lambda mz: np.full_like(np.asarray(mz, dtype=np.float64), offset)

    if model == "binned_median":
        mz = df["precursor_mz"].to_numpy()
        bin_width = config["bin_width_da"]
        min_psms_per_bin = config["min_psms_per_bin"]
        bin_idx = np.floor(mz / bin_width).astype(np.int64)
        stats = pd.DataFrame({"bin": bin_idx, "ppm": ppm}).groupby("bin")["ppm"].agg(["median", "count"])
        stats = stats[stats["count"] >= min_psms_per_bin].sort_index()
        if len(stats) < 2:
            offset = float(np.median(ppm))
            return lambda mz: np.full_like(np.asarray(mz, dtype=np.float64), offset)
        bin_centers = (stats.index.to_numpy() + 0.5) * bin_width
        bin_medians = stats["median"].to_numpy()
        return lambda mz: np.interp(mz, bin_centers, bin_medians)

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
    """Fit the ppm correction from confident PSMs, apply it to the tof2mz lookup array,
    and derive new precursor_tol/fragment_tol bounds from the (post-correction) residual
    error distributions' min/max (not a quantile -- a quantile cutoff has no safety
    margin and was found to shrink Sage's own candidate search space too aggressively,
    losing far more identifications than the tighter tolerance was worth).

    No fragment-specific model is fit -- fragment_tol is tightened using the confident
    PSMs' own observed (uncorrected) `fragment_ppm` values directly, since tof2mz is a
    single shared ToF->m/z lookup and recalibrating it improves fragments too even though
    the correction is only fit from precursor errors.
    """
    df = filter_top_psms(sage_results_tsv, fdr)
    correction = fit_correction(df, config)

    new_tof2mz = tof2mz / (1.0 + correction(tof2mz) * 1e-6)

    residual_precursor_ppm = df["precursor_ppm"].to_numpy() - correction(
        df["precursor_mz"].to_numpy()
    )
    precursor_lo, precursor_hi = residual_precursor_ppm.min(), residual_precursor_ppm.max()
    fragment_ppm = df["fragment_ppm"].to_numpy()
    fragment_lo, fragment_hi = fragment_ppm.min(), fragment_ppm.max()

    tolerance = {
        "precursor_tol": {"ppm": [float(precursor_lo), float(precursor_hi)]},
        "fragment_tol": {"ppm": [float(fragment_lo), float(fragment_hi)]},
    }
    return new_tof2mz.astype(np.float32), tolerance
