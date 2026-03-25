"""Map SAGE FDR-filtered PSMs and matched fragments back to pmsms.mmappet entries.

Mapping chain:
  scannr 'precursor_idx=N charge=C ...'
    → precursors_parquet[precursor_idx]
    → fragment_spectrum_start, fragment_event_cnt
    → pmsms_mz[start : start+cnt]
    → binary-search nearest-neighbour match on fragment_mz_experimental
    → pmsms_fragment_idx per sage matched fragment

Output: mmappet with data (pmsms_fragment_idx, mz_delta, fragment_row_idx) and
index (precursor_idx, start, count) — one data row per matched fragment.

Runtime dependencies (assumed present in venvs/common):
  mmappet, duckdb, numba, numba_progress, numpy, pandas
"""

import argparse
import numba
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

import mmappet
from numba_progress import ProgressBar
from timstofu.stats import get_index


@numba.njit(boundscheck=True)
def _first_duplicate(arr):
    """Return the first duplicate value in a sorted int64 array, or -1 if all unique."""
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            return arr[i]
    return np.int64(-1)


@numba.njit(boundscheck=True)
def _check_submitted_pairs(
    charges,  # int64[N] — submitted charges (digit-encoded) from psm_counts
    found_charges,  # int64[N] — found charges (digit-encoded) from psm_counts
    out_bad,  # bool[N]  — True where a found charge digit is not in submitted
):
    """For each row, verify every digit of found_charges is present in charges.

    Charge states are encoded as decimal digits of a single integer: e.g. charge
    states {2, 3, 4} → 234.  A digit d of found_charges[i] is "submitted" if d
    appears as a decimal digit of charges[i].  Zero is not a valid charge state
    and never appears as a digit, so the digit loop terminates correctly on the
    first division that yields 0.
    """
    n_bad = np.int64(0)
    for i in range(len(charges)):
        pc = charges[i]
        fc = found_charges[i]
        all_ok = True
        fc_tmp = fc
        while fc_tmp > np.int64(0):
            d = fc_tmp % np.int64(10)
            pc_tmp = pc
            found_d = False
            while pc_tmp > np.int64(0):
                if pc_tmp % np.int64(10) == d:
                    found_d = True
                    break
                pc_tmp //= np.int64(10)
            if not found_d:
                all_ok = False
                break
            fc_tmp //= np.int64(10)
        out_bad[i] = not all_ok
        if not all_ok:
            n_bad += np.int64(1)
    return n_bad


@numba.njit(boundscheck=True)
def _sort_exp_mz_groups(exp_mz_arr, psm_idx):
    """Sort exp_mz_arr in-place within each PSM group if not already sorted."""
    for k in range(len(psm_idx) - 1):
        gs = psm_idx[k]
        ge = psm_idx[k + 1]
        chunk = exp_mz_arr[gs:ge]
        is_sorted = True
        for i in range(len(chunk) - 1):
            if chunk[i] > chunk[i + 1]:
                is_sorted = False
                break
        if not is_sorted:
            chunk.sort()


@numba.njit(parallel=True, boundscheck=True)
def _match_fragments_numba(
    frag_start,  # int64[N_groups] — fragment_spectrum_start per PSM group
    frag_cnt,  # int64[N_groups] — fragment_event_cnt per PSM group
    psm_idx,  # int64[N_groups + 1] — CSR over exp_mz_arr: PSM k spans [psm_idx[k], psm_idx[k+1])
    pmsms_mz,  # float32[N_frags]   — sorted within each precursor slice
    exp_mz_arr,  # float32[N_rows]    — experimental fragment m/z, sorted within each PSM group
    out_idx,  # int64[N_rows]      — output: absolute pmsms fragment index
    progress=None,
):
    """Nearest-neighbour fragment matching: for each PSM group, find the closest
    pmsms library fragment for every SAGE-reported experimental m/z.

    Layout:
      - PSM group i covers exp_mz_arr[psm_idx[i] : psm_idx[i+1]] (sorted ascending).
      - The corresponding library slice is pmsms_mz[frag_start[i] : frag_start[i]+frag_cnt[i]]
        (also sorted ascending within each precursor, as written by the pipeline).

    Algorithm: O(n+m) two-pointer scan.  Pointer p advances monotonically through
    the library slice; for each query m/z we move p forward as long as the next
    library entry is strictly closer.  Ties go to the lower-m/z library entry
    (p is not advanced when distances are equal).

    Output: out_idx[j] = absolute index into pmsms_mz of the nearest library
    fragment for experimental fragment j.
    """
    for i in numba.prange(len(frag_start)):
        gs = psm_idx[i]
        ge = psm_idx[i + 1]
        start = frag_start[i]
        cnt = frag_cnt[i]
        if cnt == np.int64(0):
            if progress is not None:
                progress.update(1)
            continue
        # Two-pointer nearest-neighbour merge (both arrays sorted): O(n+m).
        # exp_mz_arr is pre-sorted within each group by the SQL query.
        # p advances monotonically; ties go to the lower m/z element.
        p = np.int64(0)
        for j in range(gs, ge):
            q = exp_mz_arr[j]
            while p + np.int64(1) < cnt and abs(
                pmsms_mz[start + p + np.int64(1)] - q
            ) < abs(pmsms_mz[start + p] - q):
                p += np.int64(1)
            out_idx[j] = start + p
        if progress is not None:
            progress.update(1)


if __name__ == "__main__":
    from pprint import pprint

    dataset = "F9477"
    cfg = "optimal2tier"
    sage_version = "devel"
    sage_cfg = "p12f15"
    fasta = "human"
    __args = dict(
        filtered_parquet=f"temp/{dataset}/{cfg}/sage/{sage_version}/{sage_cfg}/{fasta}/results/results.sage.filtered.parquet",
        matched_fragments=f"temp/{dataset}/{cfg}/sage/{sage_version}/{sage_cfg}/{fasta}/dir/matched_fragments.sage.tsv",
        precursors_parquet=f"temp/{dataset}/{cfg}/filtered_precursor_clusters_with_nontrivial_ms2.parquet",
        pmsms_dir=f"temp/{dataset}/{cfg}/pmsms.mmappet",
        output=f"/home/matteo/temp/{dataset}_{cfg}_mappedback",
        verbose=True,
        use_duckdb=True,
    )
    pprint(__args)
    locals().update(**__args)


def map_sage_to_pmsms(
    filtered_parquet: Path,
    matched_fragments: Path,
    precursors_parquet: Path,
    pmsms_dir: Path,
    output: Path,
    verbose: bool = True,
    use_duckdb: bool = True,
) -> None:
    """Map sage FDR-filtered PSMs and matched fragments to pmsms.mmappet entries.

    Writes mmappet data (pmsms_fragment_idx, mz_delta, fragment_row_idx) and
    index (precursor_idx, start, count), one data row per matched fragment.
    """
    # ── 1. Load pmsms_mz (only column needed for matching) ───────────────────
    if verbose:
        print("Loading pmsms mz...")
    pmsms_mz = mmappet.open_dataset_dct(pmsms_dir)["mz"]
    if verbose:
        print(f"  {len(pmsms_mz):_} fragment peaks")

    # ── 2. Load PSMs + matched fragments, resolve precursor slices ────────────
    if verbose:
        print("Loading sage PSMs and matched fragments...")

    if use_duckdb:
        psm_counts = duckdb.sql(
            f"""
            WITH raw AS (
                SELECT psm_id
                FROM read_csv('{matched_fragments}', sep='\\t', header=true)
            ),
            psm_map AS (
                SELECT
                    psm_id,
                    CAST(regexp_extract(scannr, 'precursor_idx=(\\d+)', 1) AS BIGINT) AS precursor_idx,
                    CAST(regexp_extract(scannr, 'charge=(\\d+)', 1) AS INTEGER)       AS charge
                FROM read_parquet('{filtered_parquet}')
            ),
            prec AS (
                SELECT precursor_idx, fragment_spectrum_start, fragment_event_cnt, charges
                FROM read_parquet('{precursors_parquet}')
                WHERE fragment_event_cnt > 0
            ),
            merged AS (
                SELECT m.precursor_idx, m.charge, p.charges,
                       p.fragment_spectrum_start, p.fragment_event_cnt
                FROM raw r
                JOIN psm_map m USING (psm_id)
                LEFT JOIN prec p ON m.precursor_idx = p.precursor_idx
            )
            SELECT
                precursor_idx,
                CAST(array_to_string(list_sort(list(DISTINCT CAST(charge AS VARCHAR))), '') AS BIGINT) AS found_charges,
                first(charges)                   AS charges,
                count(*)                         AS cnt,
                first(fragment_spectrum_start)   AS fragment_spectrum_start,
                first(fragment_event_cnt)        AS fragment_event_cnt
            FROM merged
            GROUP BY precursor_idx
            ORDER BY precursor_idx
            """
        ).df()
        unsubmitted_mask = psm_counts["charges"].isna()
        if unsubmitted_mask.any():
            bad_idx = psm_counts.loc[unsubmitted_mask, "precursor_idx"].tolist()
            raise AssertionError(
                f"\n\n💀 CATASTROPHIC PIPELINE INTEGRITY FAILURE 💀\n"
                f"SAGE returned {len(bad_idx):_} precursor_idx value(s) that do not exist in the "
                f"submitted precursors parquet.\n"
                f"The MGF was likely regenerated without rerunning SAGE. Fix it.\n\n"
                f"Offending precursor_idx: {sorted(bad_idx)[:20]}"
            )
    else:
        # Much slower, for debugging...
        # psm_id per matched fragment row (file order preserved for exp_mz_arr alignment)
        raw = duckdb.sql(
            f"SELECT psm_id FROM read_csv('{matched_fragments}', sep='\\t', header=true)"
        ).df()

        # precursor_idx and charge extracted from scannr (one row per PSM)
        psm_map = duckdb.sql(
            f"""
            SELECT
                psm_id,
                CAST(regexp_extract(scannr, 'precursor_idx=(\\d+)', 1) AS BIGINT) AS precursor_idx,
                CAST(regexp_extract(scannr, 'charge=(\\d+)', 1) AS INTEGER)       AS charge
            FROM read_parquet('{filtered_parquet}')
            """
        ).df()

        # precursor slice info (one row per precursor_idx)
        prec = duckdb.sql(
            f"""
            SELECT precursor_idx, fragment_spectrum_start, fragment_event_cnt, charges
            FROM read_parquet('{precursors_parquet}')
            WHERE fragment_event_cnt > 0
            """
        ).df()

        # one row per matched fragment, with precursor info attached.
        # Left join so unsubmitted precursor_idx values are not silently dropped.
        merged = raw.merge(psm_map, on="psm_id").merge(
            prec, on="precursor_idx", how="left"
        )[
            [
                "precursor_idx",
                "charge",
                "charges",
                "fragment_spectrum_start",
                "fragment_event_cnt",
            ]
        ]
        unsubmitted = merged[merged["charges"].isna()]["precursor_idx"].unique()
        if len(unsubmitted):
            raise AssertionError(
                f"\n\n💀 CATASTROPHIC PIPELINE INTEGRITY FAILURE 💀\n"
                f"SAGE returned {len(unsubmitted):_} precursor_idx value(s) that do not exist in the "
                f"submitted precursors parquet.\n"
                f"The MGF was likely regenerated without rerunning SAGE. Fix it.\n\n"
                f"Offending precursor_idx: {sorted(unsubmitted)[:20]}"
            )

        # aggregate to one row per precursor_idx
        psm_counts = (
            merged.groupby("precursor_idx", sort=True)
            .agg(
                found_charges=(
                    "charge",
                    lambda x: int("".join(str(c) for c in sorted(x.unique()))),
                ),
                charges=("charges", "first"),
                cnt=("charge", "count"),
                fragment_spectrum_start=("fragment_spectrum_start", "first"),
                fragment_event_cnt=("fragment_event_cnt", "first"),
            )
            .reset_index()
        )
    if verbose:
        print(f"  {len(psm_counts):_} filtered PSMs")

    dup = _first_duplicate(psm_counts["precursor_idx"].to_numpy(dtype=np.int64))
    if dup != -1:
        raise AssertionError(
            f"\n\n💀 CATASTROPHIC PIPELINE INTEGRITY FAILURE 💀\n"
            f"psm_counts.precursor_idx is not unique: first duplicate is {dup}.\n"
            f"The groupby aggregation is broken. Fix it."
        )

    # Explicit check: every found charge digit must be present in the submitted
    # charges for that precursor — both columns are already aligned in psm_counts.
    out_bad = np.zeros(len(psm_counts), dtype=np.bool_)
    n_bad = _check_submitted_pairs(
        psm_counts["charges"].to_numpy(dtype=np.int64),
        psm_counts["found_charges"].to_numpy(dtype=np.int64),
        out_bad,
    )
    if n_bad:
        bad = psm_counts[out_bad][["precursor_idx", "found_charges", "charges"]]
        raise AssertionError(
            f"\n\n💀 CATASTROPHIC PIPELINE INTEGRITY FAILURE 💀\n"
            f"SAGE returned {n_bad:_} precursor(s) with charge states that were NEVER submitted.\n"
            f"This means the scannr→precursor_idx mapping is broken, the MGF was regenerated\n"
            f"without rerunning SAGE, or some deranged developer forgot to keep the pipeline\n"
            f"inputs consistent. Whoever did this should be deeply ashamed.\n\n"
            f"Offending rows (precursor_idx | found_charges | submitted charges):\n"
            f"{bad.to_string(index=False)}"
        )

    # mz values in file order.
    exp_mz_arr = (
        duckdb.sql(
            f"SELECT fragment_mz_experimental FROM read_csv('{matched_fragments}', sep='\\t', header=true)"
        )
        .df()["fragment_mz_experimental"]
        .to_numpy(dtype=np.float32)
    )

    # CSR index: group k covers rows [group_idx[k], group_idx[k+1])
    psm_idx = get_index(psm_counts.cnt.to_numpy())

    # Sort within each PSM group if not already sorted.
    _sort_exp_mz_groups(exp_mz_arr, psm_idx)

    if verbose:
        print(f"  {len(exp_mz_arr):_} SAGE-reported fragments to match from")

    # ── 3. Nearest-neighbour match per precursor (parallel Numba kernel) ──────
    # Output: one pmsms fragment index per matched-fragment row; -1 = unmatched.
    pmsms_fragment_idx = np.full(len(exp_mz_arr), -1, dtype=np.int64)

    with ProgressBar(total=len(psm_counts), desc="Matching precursors") as progress:
        _match_fragments_numba(
            frag_start=psm_counts.fragment_spectrum_start.to_numpy(),
            frag_cnt=psm_counts.fragment_event_cnt.to_numpy(),
            psm_idx=psm_idx,
            pmsms_mz=pmsms_mz,
            exp_mz_arr=exp_mz_arr,
            out_idx=pmsms_fragment_idx,
            progress=progress,
        )

    matched = pmsms_fragment_idx != -1
    skipped = len(pmsms_fragment_idx) - matched.sum()
    if skipped and verbose:
        print(
            f"  WARNING: {skipped:_} / {len(pmsms_fragment_idx):_} = { skipped / len(pmsms_fragment_idx) * 100:.2f}% rows not matched (out-of-range precursor or empty slice)"
        )

    # ── 4. Write CSR-style mmappet output ────────────────────────────────────
    if verbose:
        print(f"Writing to {output} ...")

    matched_frag_idx = pmsms_fragment_idx[matched]
    mz_delta = pmsms_mz[matched_frag_idx] - exp_mz_arr[matched]

    # data arrays — one entry per matched fragment
    data = pd.DataFrame(
        {
            "pmsms_fragment_idx": matched_frag_idx,
            "mz_delta": mz_delta,
            "fragment_row_idx": np.where(matched)[0].astype(np.int64),
        },
        copy=False,
    )
    # index — one entry per precursor: where its fragments start and how many
    index = pd.DataFrame(
        {
            "precursor_idx": psm_counts["precursor_idx"].to_numpy(dtype=np.int64),
            "start": psm_idx[:-1],
            "count": np.diff(psm_idx),
        }
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with mmappet.DatasetWriter(output / "data.mmappet", overwrite_dir=True) as w:
        w.append_df(data)
    with mmappet.DatasetWriter(output / "index.mmappet", overwrite_dir=True) as w:
        w.append_df(index)
    if verbose:
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Map sage FDR-filtered PSMs and matched fragments to pmsms.mmappet entries."
    )
    parser.add_argument(
        "filtered_parquet", type=Path, help="results.sage.filtered.parquet"
    )
    parser.add_argument(
        "matched_fragments", type=Path, help="matched_fragments.sage.tsv"
    )
    parser.add_argument(
        "precursors_parquet",
        type=Path,
        help="filtered_precursor_clusters_with_nontrivial_ms2.parquet",
    )
    parser.add_argument("pmsms_dir", type=Path, help="pmsms.mmappet directory")
    parser.add_argument("output", type=Path, help="Output mmappet directory")
    args = parser.parse_args()

    map_sage_to_pmsms(
        filtered_parquet=args.filtered_parquet,
        matched_fragments=args.matched_fragments,
        precursors_parquet=args.precursors_parquet,
        pmsms_dir=args.pmsms_dir,
        output=args.output,
    )


if __name__ == "__main__":
    main()
