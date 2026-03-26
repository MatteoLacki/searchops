# CLAUDE.md — searchops

## Package layout

```
src/searchops/
├── sage.py                  # summarize_sage() / count_sage() — core filtering & counting
└── cli/
    ├── sage_summary.py      # sage-summary  — compare results across folders
    ├── sage_write.py        # sage-write    — export single-file summary TSV
    ├── sage_filter.py       # sage-filter   — FDR-filter TSV/parquet → parquet
    ├── tsv2parquet.py       # tsv2parquet   — convert SAGE TSV → parquet
    ├── sage_summarize.py    # sage-summarize — count pre-filtered parquet
    └── sage_pmsms_mapper.py # sage-pmsms-mapper — map fragments to mmappet library
```

## Core API (`sage.py`)

Two exported functions:
- `summarize_sage(path, fdr, level)` — filters by FDR on the fly; accepts TSV or parquet
- `count_sage(path)` — counts a pre-filtered parquet (no FDR argument); used by `sage-summarize`

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
`tsv2parquet` applies the same handling when converting to parquet.

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

## Adding a new CLI tool

1. Add `src/searchops/cli/<name>.py` with a `main()` entry point.
2. Register it in `pyproject.toml` under `[project.scripts]`.
3. Reinstall: `pip install -e .` (or `uv pip install -e .`).
