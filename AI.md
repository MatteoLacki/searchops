# CLAUDE.md — searchops

## Package layout

```
src/searchops/
├── sage.py                  # summarize_sage() / count_sage() — core filtering & counting
└── cli/
    ├── sage_summary.py      # sage-summary  — compare results across folders
    ├── sage_write.py        # sage-write    — export single-file summary TSV
    ├── sage_filter.py       # sage-filter   — FDR-filter TSV/parquet → parquet
    ├── sage_summarize_raw.py # sage-summarize-raw — filter+count raw SAGE results in one pass
    ├── sage_pmsms_mapper.py # sage-pmsms-mapper — map fragments to mmappet library
    └── sage_score_mapper.py # sage_score_mapper — visualise pmsms score distributions
```

## Core API (`sage.py`)

Two exported functions:
- `summarize_sage(path, fdr, level)` — filters by FDR on the fly; accepts TSV or parquet
- `count_sage(path)` — counts a pre-filtered parquet (no FDR argument); used by `sage-summarize-raw`

Filtering levels and their q-value columns:

| level | column |
|---|---|
| `psm` | `spectrum_q` |
| `peptide` | `peptide_q` |
| `protein` | `protein_q` |

**Ion definition**: distinct `(peptide, charge)` pairs — DuckDB `DISTINCT (peptide, charge)`.

**Protein counting**: `unnest(string_split(proteins, ';'))` expands the semicolon-delimited
protein field into one row per protein before counting distinct entries.

## Input formats

`summarize_sage()` accepts both `.tsv` and `.parquet`. DuckDB queries files
directly via `read_parquet()` / `read_csv()` — do not load into pandas first.

**TSV quirk**: SAGE's `scannr` field contains triple-quoted strings with embedded
newlines. `quote='', ignore_errors=true` in `read_csv` handles this; the rows
dropped by `ignore_errors` are the malformed ones (scannr only, not PSM data).
Every reader of raw SAGE TSV output in this package applies the same handling.

## `sage-pmsms-mapper` — fragment matching

Maps SAGE FDR-filtered PSMs and matched fragments to entries in a pmsms mmappet library.

**Inputs**:
- Filtered parquet (from `sage-filter`)
- mmappet dataset directory (pmsms library m/z array)
- Precursor slice parquet with `fragment_spectrum_start` and `fragment_event_cnt`

**Outputs** (written to one output directory):
- `precursors.parquet` — one row per matched precursor; detected/submitted charges + indices
- `mapping.parquet` — one row per matched fragment (`pmsms_fragment_idx`, `sage_fragment_idx`)
- `mz_delta_quantiles.parquet` — 101-point quantile distribution of m/z errors

**Matching algorithm**: Numba-compiled two-pointer O(n+m) scan per precursor group.
Experimental m/z values are sorted; library m/z is pre-sorted. Ties resolve to the
lower-m/z library entry (pointer is **not** advanced on equality).

**CSR indexing**: `timstofu.stats.get_index()` builds group-boundary arrays for parallel
Numba dispatch.

**Integrity checks** (fail loudly on pipeline breaks):
- All found charge states must be in submitted charges (decimal-digit-encoded: `234` → charges 2, 3, 4)
- All matched `precursor_idx` must exist in the submission parquet
- Unsubmitted `precursor_idx` raise an error

**Key dependencies**: `numba`, `timstofu`, mmappet dataset.

## `sage_score_mapper` — pmsms score visualisation

Compares the `score` column of the pmsms mmappet library between fragments matched
back by SAGE and all remaining (unmatched) fragments.

**Inputs**:
- Filtered precursor candidates parquet
- `pmsms.mmappet` directory (columns used: `score`, `intensity`)
- `sage_mapped_to_pmsms/precursors.parquet` — CSR index (`mapped_idx`, `mapped_cnt`, `detected_charges`) from `sage-pmsms-mapper`
- `sage_mapped_to_pmsms/mapping.parquet` — `pmsms_fragment_idx` per matched fragment
- Pipeline config TOML — reads `pseudomsms.tofs_extraction_method` and `tofs_extraction_params` for plot titles

**Outputs** (written to `--output` directory):
- `score_distribution.png` — 1D density histogram: matched vs unmatched overlay (matplotlib)
- `score_by_charge.png` — per-charge faceted histograms with unmatched reference in every panel (plotnine)
- `score_vs_intensity_2d.png` — side-by-side 2D heatmaps of `score` vs `log10(1+intensity)`, normalised to density (kilograms + matplotlib)
- `score_vs_intensity_isoquants.png` — Gaussian-smoothed isoquant overlay of both distributions on a dark background (matplotlib contour)

**Score label**: when `tofs_extraction_method == "score"`, the title summarises all
`tofs_extraction_params` key=value pairs wrapped at 80 characters.

**Key dependencies**: `mmappet`, `kilograms`, `plotnine`, `scipy.ndimage`.

## `recalibration.py` — mz recalibration, `[mz]`-scoped tolerance (2026-08-25)

`recalibrate_pmsms_mz`/`recalibrate_precursors` fit `config["fragment_model"]`/
`config["precursor_model"]` (unchanged, `searchops.models.build_model`) and derive
a tolerance window from the fit residual via `_select_tolerance(residual,
config["mz"])` — reads `config["mz"]["tolerance_percentiles"]` and
`config["mz"].get("tolerance_method", "theoretic")`. Previously read
`config["tolerance_percentiles"]` off the config root; moved into its own `[mz]`
table so mz's percentiles/method are independently configurable from RT/IIM's
(`git/featureprediction`'s own `[rt]`/`[iim]` equivalents), not implicitly shared
just by sitting at the same nesting level. `fragment_model`/`precursor_model`
stay at the config root — only `tolerance_percentiles`/`tolerance_method` moved.
Design/history: `plans/lda_external_rt_iim_features.md`.

`"theoretic"` (default) — `_symmetric_tolerance`, `median ± z·robust_sigma`
(`z = norm.ppf(hi_pct/100)`), mirrors `feature_prediction.tolerance.
symmetric_gaussian_tolerance` exactly (separate implementation, not a shared
import — different package). `"empiric"` — `_tolerance`, the original plain
empirical percentiles (kept, still used when explicitly selected). Default
changed to theoretic because real F9477 precursor mass-error residuals are
visibly right-skewed, mostly a truncation artifact of the calibration pass's
own fixed search window rather than genuine distribution shape worth chasing
with an asymmetric window.

**`tests/test_recalibration.py` is currently broken independent of this
change** — found while adding tests here, not caused by it: it imports
`recalibrate` from this module, which doesn't exist (only
`recalibrate_pmsms_mz`/`recalibrate_precursors` do — likely stale from a prior
split/rename this test file never followed). Left as-is; new coverage for the
tolerance-selection change lives in `tests/test_tolerance_selection.py` instead.

## Adding a new CLI tool

1. Add `src/searchops/cli/<name>.py` with a `main()` entry point.
2. Register it in `pyproject.toml` under `[project.scripts]`.
3. Reinstall: `pip install -e .` (or `uv pip install -e .`).
