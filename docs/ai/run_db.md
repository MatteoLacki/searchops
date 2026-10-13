# Run database (`run_db.py`, `record-run`)

A SQLite file at `$ION_MAIDEN_REGRESSION_DB` with one row per SAGE or mokapot pipeline
node that ran. Nothing is recorded when the variable is unset.

## Who writes it

The last step of the pipeline's `run_sage` and `mokapot` commands (necromerge2
`git/ionmaidentools/src/ionmaidentools/pipelines.py`) is `sage-fdr-levels` /
`mokapot-fdr-levels` (`cli/fdr_levels.py`). It writes the node's FDR-levels TSV, then
calls `run_db.record_run` if the variable is set. A failed record prints
`not recorded in …` to the node's `job.log` and never fails the node.

A row is written only when the node actually runs. A cached node records nothing, and
setting the variable afterwards does not record runs that already happened. Every
`run_sage` node records, including the recalibration pass (`filtered_sage_*` labels),
so one pipeline run of `jobs/f9477_best.toml` gives two `sage` rows and one `mokapot` row.

## Schema: table `runs`

| column | content |
|---|---|
| `id` | row id, used by `record-run comment` |
| `recorded_at` | UTC time the row was written, at the end of the node's command |
| `tool` | `sage` or `mokapot` |
| `node` | node directory relative to the pipeline checkout, `nodes/<rule>/<hash>` |
| `pipeline_root` | absolute path of the pipeline checkout |
| `dataset` | `source_bruker_d`'s `path` among the node's ancestors, e.g. `data/F9477.d` |
| `command` | the tool command: the SAGE call, or the pin adapter + mokapot call |
| `config` | JSON `{"<rule>/<hash>": {"rule", "config", "command"}}` for every node upstream |
| `git` | JSON `[{"path", "branch", "commit", "dirty"}]` for the checkout (`.`) and each repo in `git/` |
| `fdr` | FDR threshold of the counts (0.01) |
| `spectra`, `peptidoforms`, `ions` | confident target counts, each from its own competition (`docs/ai/ion_level_fdr.md`) |
| `comment`, `commented_at` | free text added after the run, and when |

`config` holds each ancestor's own config. Necroflow's `.rip/dependencies.toml` stores a
node's config merged over all its ancestors' (nearest wins), so `ancestor_nodes` keeps
only the keys named by the node's rule's scalar parameters (`[recipe.input_types]` of
`builtins.*` types). Text-file nodes (SAGE config, recalibration config, …) carry their
whole text this way. It is about 40 KB per row for F9477.

The command is passed in rather than read from the node: necroflow writes a node's
`.rip/dependencies.toml` only after its command finishes, so the recorder reads its
inputs' metadata (`--parent`, one per input path) and walks up from there.

`git` is taken when the node finishes, from the working tree, so it says what was
checked out then, which is not necessarily what built the binaries. `dirty` counts
tracked files only.

Concurrent writers: WAL journal and a 60 s busy timeout.

## Using it

```bash
export ION_MAIDEN_REGRESSION_DB=/home/matteo/Projects/ionmaiden/regression.sqlite  # set in ~/.bashrc
venvs/common/bin/record-run init             # create the file and schema (make init-regression-db)
venvs/common/bin/record-run list -n 20       # latest rows, without config/git
venvs/common/bin/record-run comment 7 "text" # set a run's comment
```

From Python, `json.loads` the `config`/`git` columns, e.g. to diff two runs' configs.

An AI agent that checks or compares configs reads the latest rows for the dataset and
their comments before judging a change, and after a run it writes a short `comment`:
what the run tested and how its counts compare with the runs it should be compared to.

## History

Until 2026-10-13 the file at that path held the Snakemake-era schema (`runs` with
psm/peptide/ion/protein counts, `git_snapshots`, review markers), 122 rows from
2026-04-15 to 2026-07-07, written by the old `record-run`. It is kept as
`/home/matteo/Projects/ionmaiden/regression_snakemake_until_2026-10-13.sqlite`; the
old recorder and its tests were replaced. `scripts/export_runs_to_parquet.py` in
necromerge2 reads that old schema.
