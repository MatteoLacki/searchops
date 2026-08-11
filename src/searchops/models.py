"""Pluggable ppm-correction models: fit/predict, self-serializing, self-plotting.

Every concrete model reduces, once fitted, to a plain array-in/array-out
function -- so `save`/`load` never need to pickle a live scipy/xgboost/spline
object, just the small arrays that fully determine `predict`.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mmappet
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, CubicSpline, PPoly
from xgboost import XGBRegressor

from searchops.recalibration import _derivative_penalized_smooth, _fit_pspline


def _dump_arrays(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=np.float64)
        ds = mmappet.open_new_dataset_dct(
            path / f"{name}.mmappet", scheme=mmappet.get_schema(**{name: np.float64}), nrows=arr.shape[0],
        )
        ds[name][:] = arr


def _load_arrays(path: Path, *names: str) -> tuple[np.ndarray, ...]:
    path = Path(path)
    out = []
    for name in names:
        ds = mmappet.open_dataset_dct(path / f"{name}.mmappet")
        out.append(np.asarray(ds[name], dtype=np.float64))
    return tuple(out)


def _dump_scalars(path: Path, **scalars: float) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ds = mmappet.open_new_dataset_dct(
        path / "scalars.mmappet",
        scheme=mmappet.get_schema(**{k: np.float64 for k in scalars}),
        nrows=1,
    )
    for k, v in scalars.items():
        ds[k][0] = float(v)


def _load_scalars(path: Path, *names: str) -> tuple[float, ...]:
    ds = mmappet.open_dataset_dct(Path(path) / "scalars.mmappet")
    return tuple(float(ds[name][0]) for name in names)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Smallest value where cumulative weight reaches half the total."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = weights.sum() / 2.0
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cutoff)])


class MzCorrectionModel(ABC):
    """A fittable, self-serializing, self-plotting ppm-vs-m/z correction."""

    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> "MzCorrectionModel":
        return self

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "MzCorrectionModel": ...

    def plot_fit(
        self, path: str | Path, x: np.ndarray, y: np.ndarray,
        weight: np.ndarray | None = None, title: str | None = None,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.asarray(x)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(x, y, s=3, alpha=0.2, color="#0072B2", linewidths=0, label=f"data (n={len(x):_})")
        line_x = np.linspace(x.min(), x.max(), 400)
        ax.plot(line_x, self.predict(line_x), color="black", linewidth=2.5, label="fitted correction")
        ax.axhline(0, color="#808080", linewidth=1, linestyle=":")
        ax.set_xlabel("m/z")
        ax.set_ylabel("ppm error")
        ax.legend(fontsize=9, loc="best")
        if title:
            ax.set_title(title)
        fig.tight_layout()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)


class GlobalMedianModel(MzCorrectionModel):
    """A single constant correction: the (weighted) median of `y`."""

    def __init__(self) -> None:
        self.value = 0.0

    def fit(self, x, y, weight=None) -> "GlobalMedianModel":
        y = np.asarray(y)
        self.value = float(np.median(y)) if weight is None else _weighted_median(y, np.asarray(weight))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=np.float64), self.value)

    def save(self, path: str | Path) -> None:
        _dump_scalars(path, value=self.value)

    @classmethod
    def load(cls, path: str | Path) -> "GlobalMedianModel":
        (value,) = _load_scalars(path, "value")
        model = cls()
        model.value = value
        return model


class BinnedMedianModel(MzCorrectionModel):
    """m/z-binned (weighted) median, `np.interp`-ed between bin centers,
    constant beyond the outermost bins."""

    def __init__(self, bin_width_da: float) -> None:
        self.bin_width_da = bin_width_da
        self.bin_centers = np.zeros(0)
        self.bin_medians = np.zeros(0)

    def fit(self, x, y, weight=None) -> "BinnedMedianModel":
        x, y = np.asarray(x), np.asarray(y)
        bin_idx = np.floor(x / self.bin_width_da).astype(np.int64)
        if weight is None:
            medians = pd.DataFrame({"bin": bin_idx, "y": y}).groupby("bin")["y"].median().sort_index()
        else:
            medians = (
                pd.DataFrame({"bin": bin_idx, "y": y, "weight": weight})
                .groupby("bin")
                .apply(lambda g: _weighted_median(g["y"].to_numpy(), g["weight"].to_numpy()))
                .sort_index()
            )
        self.bin_centers = (medians.index.to_numpy() + 0.5) * self.bin_width_da
        self.bin_medians = medians.to_numpy(dtype=np.float64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.bin_centers, self.bin_medians)

    def save(self, path: str | Path) -> None:
        _dump_arrays(path, bin_centers=self.bin_centers, bin_medians=self.bin_medians)
        _dump_scalars(path, bin_width_da=self.bin_width_da)

    @classmethod
    def load(cls, path: str | Path) -> "BinnedMedianModel":
        (bin_width_da,) = _load_scalars(path, "bin_width_da")
        model = cls(bin_width_da)
        model.bin_centers, model.bin_medians = _load_arrays(path, "bin_centers", "bin_medians")
        return model


class PSplineModel(MzCorrectionModel):
    """Penalized B-spline (Eilers & Marx 1996) fit via `recalibration._fit_pspline`,
    boundary-flattened, clamped to its own fitted domain outside `[lo, hi]`."""

    def __init__(self, bin_width_da: float, lam1: float = 0.0, lam2: float = 0.0, degree: int = 3) -> None:
        self.bin_width_da = bin_width_da
        self.lam1 = lam1
        self.lam2 = lam2
        self.degree = degree
        self.spline: BSpline | None = None
        self.lo = 0.0
        self.hi = 0.0

    def fit(self, x, y, weight=None) -> "PSplineModel":
        x, y = np.asarray(x), np.asarray(y)
        w = np.ones_like(y) if weight is None else np.asarray(weight)
        self.spline, self.lo, self.hi = _fit_pspline(x, y, w, self.bin_width_da, self.lam1, self.lam2, self.degree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.spline(np.clip(np.asarray(x, dtype=np.float64), self.lo, self.hi))

    def save(self, path: str | Path) -> None:
        t, c, k = self.spline.t, self.spline.c, self.spline.k
        _dump_arrays(path, knots=t, coeffs=c)
        _dump_scalars(path, degree=k, lo=self.lo, hi=self.hi, bin_width_da=self.bin_width_da, lam1=self.lam1, lam2=self.lam2)

    @classmethod
    def load(cls, path: str | Path) -> "PSplineModel":
        degree, lo, hi, bin_width_da, lam1, lam2 = _load_scalars(
            path, "degree", "lo", "hi", "bin_width_da", "lam1", "lam2"
        )
        knots, coeffs = _load_arrays(path, "knots", "coeffs")
        model = cls(bin_width_da, lam1, lam2, int(degree))
        model.spline = BSpline(knots, coeffs, int(degree), extrapolate=False)
        model.lo, model.hi = lo, hi
        return model


class NaturalCubicSplineModel(MzCorrectionModel):
    """Cubic spline through wide-bin medians, flat first derivative at both
    ends, clamped to its own fitted domain outside `[lo, hi]`."""

    def __init__(self, bin_width_da: float) -> None:
        self.bin_width_da = bin_width_da
        self.ppoly: PPoly | None = None
        self.lo = 0.0
        self.hi = 0.0

    def fit(self, x, y, weight=None) -> "NaturalCubicSplineModel":
        x, y = np.asarray(x), np.asarray(y)
        n_bins = len(np.unique(np.floor(x / self.bin_width_da).astype(np.int64)))
        n_knots = max(4, n_bins // 3)
        wide_bin_width = (x.max() - x.min()) / n_knots
        wide_bin_idx = np.floor((x - x.min()) / wide_bin_width).astype(np.int64)
        if weight is None:
            nodes = pd.DataFrame({"bin": wide_bin_idx, "x": x, "y": y}).groupby("bin").median().sort_values("x")
        else:
            grouped = pd.DataFrame({"bin": wide_bin_idx, "x": x, "y": y, "weight": weight}).groupby("bin")
            nodes = pd.DataFrame({
                "x": grouped["x"].median(),
                "y": grouped.apply(lambda g: _weighted_median(g["y"].to_numpy(), g["weight"].to_numpy())),
            }).sort_values("x")
        node_x, node_y = nodes["x"].to_numpy(), nodes["y"].to_numpy()
        spline = CubicSpline(node_x, node_y, bc_type=((1, 0.0), (1, 0.0)))
        self.ppoly = PPoly(spline.c, spline.x, extrapolate=False)
        self.lo, self.hi = float(node_x.min()), float(node_x.max())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ppoly(np.clip(np.asarray(x, dtype=np.float64), self.lo, self.hi))

    def save(self, path: str | Path) -> None:
        c = self.ppoly.c
        _dump_arrays(path, breakpoints=self.ppoly.x, coeffs_flat=c.reshape(-1))
        _dump_scalars(path, n_intervals=c.shape[1], lo=self.lo, hi=self.hi, bin_width_da=self.bin_width_da)

    @classmethod
    def load(cls, path: str | Path) -> "NaturalCubicSplineModel":
        n_intervals, lo, hi, bin_width_da = _load_scalars(path, "n_intervals", "lo", "hi", "bin_width_da")
        breakpoints, coeffs_flat = _load_arrays(path, "breakpoints", "coeffs_flat")
        c = coeffs_flat.reshape(4, int(n_intervals))
        model = cls(bin_width_da)
        model.ppoly = PPoly(c, breakpoints, extrapolate=False)
        model.lo, model.hi = lo, hi
        return model


class XGBoostDerivativePenalizedModel(MzCorrectionModel):
    """Node values from an `XGBRegressor` fit, smoothed by a derivative-penalized
    solve, `np.interp`-ed between nodes. The booster itself is discarded after
    fitting -- `predict` only ever needs the smoothed node grid."""

    def __init__(self, bin_width_da: float, lam: float, xgboost_kwargs: dict[str, Any] | None = None) -> None:
        self.bin_width_da = bin_width_da
        self.lam = lam
        self.xgboost_kwargs = xgboost_kwargs or {}
        self.bin_centers = np.zeros(0)
        self.smoothed = np.zeros(0)

    def fit(self, x, y, weight=None) -> "XGBoostDerivativePenalizedModel":
        x, y = np.asarray(x), np.asarray(y)
        kwargs = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "reg_lambda": 1.0, **self.xgboost_kwargs}
        regressor = XGBRegressor(**kwargs)
        regressor.fit(x.reshape(-1, 1), y, sample_weight=weight)

        bin_idx = np.floor(x / self.bin_width_da).astype(np.int64)
        counts = pd.Series(bin_idx).value_counts().sort_index()
        self.bin_centers = (counts.index.to_numpy() + 0.5) * self.bin_width_da
        node_y = regressor.predict(self.bin_centers.reshape(-1, 1))
        self.smoothed = _derivative_penalized_smooth(node_y, counts.to_numpy(dtype=np.float64), self.lam)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.bin_centers, self.smoothed)

    def save(self, path: str | Path) -> None:
        _dump_arrays(path, bin_centers=self.bin_centers, smoothed=self.smoothed)
        _dump_scalars(path, bin_width_da=self.bin_width_da, lam=self.lam)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostDerivativePenalizedModel":
        bin_width_da, lam = _load_scalars(path, "bin_width_da", "lam")
        model = cls(bin_width_da, lam)
        model.bin_centers, model.smoothed = _load_arrays(path, "bin_centers", "smoothed")
        return model


def build_model(config: dict) -> MzCorrectionModel:
    """Instantiate `config["class"]` (dotted path) with `config.get("kwargs", {})`."""
    module_name, class_name = config["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**config.get("kwargs", {}))
