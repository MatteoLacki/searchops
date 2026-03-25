"""Show that LEFT JOIN in DuckDB preserves unmatched rows as NULLs.

Run with:
    python tests/test_left_join_nans.py
"""

import duckdb
import numpy as np

result = duckdb.sql("""
    WITH psm_map AS (
        SELECT * FROM (VALUES
            (1, 10, 2),
            (2, 99, 3),   -- precursor_idx=99 does not exist in prec
            (3, 30, 2)
        ) t(psm_id, precursor_idx, charge)
    ),
    prec AS (
        SELECT * FROM (VALUES
            (10, 2,  0,   5),
            (30, 23, 100, 8)
        ) t(precursor_idx, charges, fragment_spectrum_start, fragment_event_cnt)
    )
    SELECT m.psm_id, m.precursor_idx, m.charge,
           p.charges, p.fragment_spectrum_start, p.fragment_event_cnt
    FROM psm_map m
    LEFT JOIN prec p USING (precursor_idx)
    ORDER BY m.psm_id
""").df()

print(result)
print()

unsubmitted = result[result["charges"].isna()]["precursor_idx"].unique()
assert len(unsubmitted) == 1 and unsubmitted[0] == 99, "expected precursor_idx=99 to be unmatched"
print(f"Unmatched precursor_idx: {unsubmitted}  ✓")
