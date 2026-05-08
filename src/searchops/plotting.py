import numpy as np
import numba
import pandas as pd
import plotnine as P


@numba.njit
def _q_sorted(arr, q):
    n = len(arr)
    pos = q * (n - 1)
    lo = int(pos)
    hi = lo + 1
    if hi >= n:
        return float(arr[n - 1])
    return float(arr[lo]) + (pos - lo) * float(arr[hi] - arr[lo])


@numba.njit(parallel=True)
def bin_box_stats(sorted_vals, bin_starts, bin_ends, n_bins):
    """Per-bin five-number summary: [min, Q1, median, Q3, max].

    Parameters
    ----------
    sorted_vals : float64 array, values sorted by bin then by value within bin
    bin_starts  : int64 array, shape (n_bins,), start offset into sorted_vals
    bin_ends    : int64 array, shape (n_bins,), exclusive end offset
    n_bins      : int

    Returns
    -------
    out : float64 array, shape (n_bins, 5); NaN row when bin is empty.
    """
    out = np.full((n_bins, 5), np.nan)
    for i in numba.prange(n_bins):
        s, e = bin_starts[i], bin_ends[i]
        if e <= s:
            continue
        arr = sorted_vals[s:e]
        out[i, 0] = arr[0]
        out[i, 1] = _q_sorted(arr, 0.25)
        out[i, 2] = _q_sorted(arr, 0.50)
        out[i, 3] = _q_sorted(arr, 0.75)
        out[i, 4] = arr[-1]
    return out


def _bin_stats_for_group(x, y, bin_edges, n_bins, labels):
    bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, n_bins - 1).astype(np.int64)
    order = np.lexsort((y, bin_idx))
    s_bins = bin_idx[order].astype(np.int64)
    s_vals = y[order].astype(np.float64)
    starts = np.searchsorted(s_bins, np.arange(n_bins, dtype=np.int64)).astype(np.int64)
    ends = np.searchsorted(s_bins, np.arange(n_bins, dtype=np.int64), side="right").astype(np.int64)
    stats = bin_box_stats(s_vals, starts, ends, n_bins)
    return [
        {"bin": labels[i],
         "ymin": stats[i, 0], "lower": stats[i, 1], "middle": stats[i, 2],
         "upper": stats[i, 3], "ymax": stats[i, 4]}
        for i in range(n_bins) if not np.isnan(stats[i, 2])
    ]


def make_kebab_df(x_m, y_m, x_u, y_u, n_bins=100):
    """Build precomputed boxplot stats DataFrame for two groups over equal-width bins.

    Parameters
    ----------
    x_m, x_u : float arrays — x-axis values for matched / unmatched groups
    y_m, y_u : float arrays — y-axis values (e.g. score_delta)
    n_bins    : number of equal-width bins over the combined x range

    Returns
    -------
    DataFrame with columns: bin (Categorical), group, ymin, lower, middle, upper, ymax
    """
    x_all = np.concatenate([x_m, x_u])
    bin_edges = np.linspace(x_all.min(), x_all.max(), n_bins + 1)
    labels = np.array([f"{(bin_edges[i] + bin_edges[i+1]) / 2:.3f}" for i in range(n_bins)])
    recs_m = _bin_stats_for_group(x_m, y_m, bin_edges, n_bins, labels)
    recs_u = _bin_stats_for_group(x_u, y_u, bin_edges, n_bins, labels)
    for r in recs_m:
        r["group"] = "matched"
    for r in recs_u:
        r["group"] = "unmatched"
    df = pd.DataFrame(recs_m + recs_u)
    df["bin"] = pd.Categorical(df["bin"], categories=labels.tolist(), ordered=True)
    return df


def kebab_plot(df, x_label, y_label, title):
    """Plotnine boxplot from precomputed stats (stat='identity', no outliers).

    Parameters
    ----------
    df      : output of make_kebab_df
    x_label : label for the x-axis (bin variable name)
    y_label : label for the y-axis
    title   : plot title

    Returns
    -------
    plotnine ggplot object
    """
    return (
        P.ggplot(df, P.aes(x="bin", fill="group",
                           ymin="ymin", lower="lower", middle="middle",
                           upper="upper", ymax="ymax"))
        + P.geom_boxplot(stat="identity", position=P.position_dodge(width=0.8))
        + P.coord_flip()
        + P.scale_fill_manual(values={"matched": "steelblue", "unmatched": "firebrick"})
        + P.labs(x=x_label, y=y_label, title=title)
        + P.theme_bw()
        + P.theme(figure_size=(12, 20))
    )
