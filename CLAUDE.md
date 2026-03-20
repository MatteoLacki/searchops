# CLAUDE.md — searchops

## Package layout

```
src/searchops/
├── sage.py          # summarize_sage() — core parsing logic
└── cli/
    ├── sage_summary.py   # sage-summary CLI (compare across folders)
    └── sage_write.py     # sage-write CLI (dump summary TSV)
```

## Input formats

`summarize_sage()` accepts both `.tsv` and `.parquet`. DuckDB queries files
directly via `read_parquet()` / `read_csv()` — do not load into pandas first.

TSV quirk: SAGE's `scannr` field contains triple-quoted strings with embedded
newlines. `quote='', ignore_errors=true` in `read_csv` handles this; the rows
dropped by `ignore_errors` are the malformed ones (scannr only, not PSM data).

## Adding a new CLI tool

1. Add `src/searchops/cli/<name>.py` with a `main()` entry point.
2. Register it in `pyproject.toml` under `[project.scripts]`.
3. Reinstall: `pip install -e .` (or `uv pip install -e .`).
