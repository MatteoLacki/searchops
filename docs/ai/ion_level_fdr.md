# Ion-level FDR (`ion_level_fdr.py`)

Counts confident target peptides and ions the way DIA-NN controls FDR for precursors. An
ion is a `(peptide, charge)` pair. Each ion is scored by its best PSM, target and decoy
ions are ranked together by that score, and target-decoy competition gives each ion a
q-value. The confident peptides are the distinct peptides of the ions at q ≤ `fdr`.

| function | inputs | score | target flag |
|---|---|---|---|
| `sage_ion_level_ids(results_tsv, fdr)` | `results.sage.tsv`, rank-1 rows only | `sage_discriminant_score` | `label == 1` |
| `mokapot_ion_level_ids(psms_txt, decoy_psms_txt, sage_results_tsv, fdr)` | `mokapot.psms.txt` + `mokapot.decoy.psms.txt` | `mokapot score` | `Label == True` |

Both return `(peptides, ions)` as sets. mokapot reports no charge, so each mokapot PSM
takes it from the SAGE results it was rescored from (`SpecId` = `psm_id`); the join
raises if a SpecId is missing or its peptide differs.

Everything is DuckDB SQL reading the result tables directly (`read_table`: `read_csv`
tab-separated, or `read_parquet`). `qvalues_query(scored)` is the q-value estimate over
any query with `score` and boolean `is_target`: FDR = (decoys + 1) / targets at each score
threshold, counted by a `RANGE ... CURRENT ROW` window ordered by score, so tied scores
share the counts of their whole tie block; q is a second window's running minimum of the
FDR from the lowest score up, capped at 1. `tdc_qvalues(scores, is_target)` runs it on
arrays. On F9477's 361,279 SAGE scores (6,215 of them tied) it equals
`mokapot.qvalues.tdc` exactly (checked against an earlier numpy version that matched
mokapot; the SQL version gives bit-identical q-values, 2026-10-13).

## mokapot needs `--keep_decoys`

mokapot writes `mokapot.decoy.psms.txt` only with `--keep_decoys`. The pipeline's
`mokapot` rule passes it since 2026-10-13, so mokapot nodes from then on hold the decoy
PSMs; older nodes don't. With the flag, the target files (`mokapot.psms.txt`,
`mokapot.peptides.txt`) are byte-identical to a run without it (checked on F9477, seed
1, xgboost plugin, both by hand and through the pipeline).

mokapot's own `LinearPsmDataset.assign_confidence` gives the same ion-level numbers when
its `peptide_column` is an ion column (`peptide/charge`): its peptide level is the best
PSM per value of that column followed by target-decoy competition, and the column is
used nowhere in training. Checked on the baseline below: 30,568 ions, 28,281 peptides.

## The counting definitions in use

Each programme and each way of counting has its own function; they are deliberately not
one code path with adapters.

| programme | definition | function |
|---|---|---|
| SAGE | peptides and ions of rank-1 target PSMs with `peptide_q` ≤ fdr | `sage.count_sage_at_fdr` |
| SAGE | ion-level, as above | `ion_level_fdr.sage_ion_level_ids` |
| mokapot | peptides from `mokapot.peptides.txt` (peptide-level q); ions from `mokapot.psms.txt` (PSM-level q) | `pipeline_analysis.raw_search.report.mokapot_ids` |
| mokapot | ion-level, as above | `ion_level_fdr.mokapot_ion_level_ids` |

The two mokapot numbers of `mokapot_ids` come from different files at different FDR
levels: its ions belong to more peptides than it reports (31,540 against 28,041 on the
F9477 baseline below). Its ions are PSMs at 1% PSM-level FDR collapsed to ions; on that
baseline 870 decoy ions score above the same cutoff against 34,066 target ions, an
ion-level FDR of about 2.6%, because a true ion has about 2.9 PSMs and a false one about
1.1.

## What the pipeline reports (since 2026-10-13)

`fdr_levels.py`: three counts per SAGE or mokapot node, each at 1% FDR from its own
competition, and no stripped-sequence count.

| level | SAGE (`sage_fdr_levels`) | mokapot (`mokapot_fdr_levels`) |
|---|---|---|
| spectra | rank-1 targets, `spectrum_q` | `mokapot.psms.txt` q-value (one PSM per spectrum) |
| peptidoforms | distinct `peptide` of rank-1 targets, `peptide_q` | `mokapot.peptides.txt` q-value |
| ions | `sage_ion_level_ids` | `mokapot_ion_level_ids` |

A peptidoform is the peptide string as SAGE writes it, modifications included. A
spectrum is one of our pseudo-spectra (one per MS1 precursor cluster), so spectrum
counts are not comparable with other tools'. Recomputing mokapot's spectrum and
peptide levels from its target and decoy scores with `tdc_qvalues` gives exactly its own
counts (98,875 and 28,041 on the baseline). DIA-NN's precursors are our ions; DIA-NN
reports no peptide count, and the peptide numbers people quote for it are the distinct
sequences of its 1% precursors.

DIA-NN (2.5.1, F9477, `~/Projects/ionmaiden/dianncomparison/F9477/report.parquet`)
filters `Q.Value` ≤ 0.01, its run-specific precursor q-value, and its precursor is our
ion: 40,166 precursors, 36,884 distinct peptides. It applies no peptide-level filter
(`Peptidoform.Q.Value` is 0 throughout that report).

## F9477 baseline

Grid cell `urt_scan__cosine` of necromerge2 `jobs/f9477_marginal_ms1_grid.toml`
(SAGE node `run_sage/8231018f…`, mokapot node `mokapot/ad3a0af8…`, 2026-10-12):

| definition | peptides | ions |
|---|---|---|
| SAGE, `count_sage_at_fdr` | 21,316 | 24,838 |
| SAGE, ion-level | 21,627 | 23,356 |
| mokapot, `mokapot_ids` | 28,041 | 34,066 |
| mokapot, ion-level | 28,281 | 30,568 |

The pipeline's three levels on the same nodes:

| | spectra | peptidoforms | ions |
|---|---|---|---|
| SAGE | 83,370 | 21,316 | 23,356 |
| mokapot | 98,875 | 28,041 | 30,568 |
