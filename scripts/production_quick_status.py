import json, csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
manifest_path = REPO_ROOT / "data" / "cloud_runs" / "production_manifest.jsonl"

latest_by_stem = {}
with manifest_path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        latest_by_stem[rec["paper_stem"]] = rec

with_rows = [r for r in latest_by_stem.values() if r.get("outcome_category") == "completed_with_dataset_rows"]
print(f"papers marked completed_with_dataset_rows: {len(with_rows)}")

total_rows = 0
uts_nonnull = ys_nonnull = elong_nonnull = 0
for r in with_rows:
    run_id = r["run_id"]
    ds_path = REPO_ROOT / "data" / "cloud_runs" / run_id / "dataset" / "cpp_dataset_v2.csv"
    if not ds_path.is_file():
        print(f"  WARN: missing dataset csv for {r['paper_stem']} ({run_id})")
        continue
    with ds_path.open() as f:
        rows = list(csv.DictReader(f))
    total_rows += len(rows)
    for row in rows:
        if row.get("UTS_MPa") not in (None, ""):
            uts_nonnull += 1
        if row.get("YS_MPa") not in (None, ""):
            ys_nonnull += 1
        if row.get("elongation_pct") not in (None, ""):
            elong_nonnull += 1

print(f"total final dataset rows emitted so far: {total_rows}")
print(f"UTS_MPa non-null: {uts_nonnull}")
print(f"YS_MPa non-null: {ys_nonnull}")
print(f"elongation_pct non-null: {elong_nonnull}")
