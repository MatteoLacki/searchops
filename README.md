# searchops

Python tools for parsing and summarizing [SAGE](https://github.com/lazear/sage) proteomics search engine results.

## Installation

```bash
pip install -e .
```

## Library

```python
from searchops.sage import summarize_sage

# works with both .tsv and .parquet
counts = summarize_sage("results.sage.tsv", fdr=0.01, level="peptide")
# {'psm_count': ..., 'peptide_count': ..., 'ion_count': ..., 'protein_count': ...}
```

`level` selects which q-value gates the filter:

| `level`    | column filtered |
|------------|----------------|
| `psm`      | `spectrum_q`   |
| `peptide`  | `peptide_q`    |
| `protein`  | `protein_q`    |

`ion_count` is defined as unique `(peptide, charge)` pairs.

## CLI tools

### `sage-summary` — compare results across folders

```bash
sage-summary [OPTIONS] FOLDER [FOLDER ...]
```

Finds all `results.sage.tsv` (or `.parquet`) files recursively within each
folder and prints a CSV comparison table sorted by the selected level count.

```
Options:
  --fdr FLOAT              FDR threshold (default: 0.01)
  --level {psm,peptide,protein}
  --glob PATTERN           file pattern within each folder
                           (default: **/results.sage.tsv)
  --output {table,csv,json}
```

Shell-expanded globs also work:

```bash
sage-summary temp/F9468/**/results.sage.tsv
```

### `sage-write` — write a summary TSV for one or more files

```bash
sage-write [OPTIONS] FILE [FILE ...]
```

Writes one row per input file to stdout (default) or `--output PATH`.

```
Options:
  --fdr FLOAT
  --level {psm,peptide,protein}
  --output PATH            file to write to (default: stdout)
```

Example — pipe into a file:

```bash
sage-write results.sage.tsv > summary.tsv
sage-write results.sage.parquet --output summary.tsv
```
