"""List how much every output table moved between the original experiment and the anchor-fix rerun.

For each CSV under outputs/ present in both experiments, rows are aligned on their text columns and every
numeric column is compared. The report gives, per file, the largest absolute change and the column and row where
it happens, so the manuscript numbers that need updating can be found without guessing.

Usage: python compare_outputs_before_after.py
Output: ../outputs/ANCHORFIX_OLD_NEW_DIFF.csv and ../outputs/ANCHORFIX_OLD_NEW_DIFF.txt
"""
import glob
import os

import numpy as np
import pandas as pd

EXPERIMENTS = "project/Experiments/"
OLD = EXPERIMENTS + "EXTEND_2026_AUG31_EVENTS_20260918/outputs"
NEW = EXPERIMENTS + "EXTEND_2026_AUG31_ANCHORFIX_20260924/outputs"

rows = []
for new_path in sorted(glob.glob(os.path.join(NEW, "**", "*.csv"), recursive=True)):
    relative = os.path.relpath(new_path, NEW)
    old_path = os.path.join(OLD, relative)
    if relative.startswith("ANCHORFIX_") or not os.path.isfile(old_path):
        continue
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    keys = [c for c in old.columns if c in new.columns and old[c].dtype == object]
    numeric = [c for c in old.columns if c in new.columns and np.issubdtype(old[c].dtype, np.number)
               and np.issubdtype(new[c].dtype, np.number)]
    if keys:
        old = old.drop_duplicates(keys)
        new = new.drop_duplicates(keys)
        merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    elif len(old) == len(new):
        merged = old.add_suffix("_old").join(new.add_suffix("_new"))
        merged["_merge"] = "both"
    else:
        rows.append({"file": relative, "rows_old": len(old), "rows_new": len(new), "note": "row count differs"})
        continue
    unmatched = int((merged["_merge"] != "both").sum())
    largest, where = 0.0, ""
    for column in numeric:
        change = (merged[column + "_new"] - merged[column + "_old"]).abs()
        if change.notna().any() and change.max() > largest:
            largest = float(change.max())
            row = merged.loc[change.idxmax()]
            where = column + " @ " + " / ".join(str(row[k]) for k in keys[:3])
    rows.append({"file": relative, "rows_old": len(old), "rows_new": len(new), "unmatched_rows": unmatched,
                 "largest_change": largest, "where": where})

report = pd.DataFrame(rows)
report.to_csv(os.path.join(NEW, "ANCHORFIX_OLD_NEW_DIFF.csv"), index=False)
with open(os.path.join(NEW, "ANCHORFIX_OLD_NEW_DIFF.txt"), "w", encoding="utf-8") as handle:
    handle.write(report.to_string(index=False) + "\n")
print(report.to_string(index=False))
