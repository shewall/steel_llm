import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "datasets_v2" / "production_run"

with (OUT_DIR / "cpp_dataset_v2.csv").open() as f:
    rows = list(csv.DictReader(f))

print(f"TOTAL ROWS: {len(rows)}")
dois = [r["doi"] for r in rows]
unique_dois = set(dois)
print(f"UNIQUE DOIs (independent papers): {len(unique_dois)}")

for target in ("UTS_MPa", "YS_MPa", "elongation_pct"):
    target_rows = [r for r in rows if r.get(target) not in (None, "")]
    target_dois = {r["doi"] for r in target_rows}
    print(f"{target}: {len(target_rows)} observations, {len(target_dois)} unique DOIs")

print("\nROWS PER PAPER (distribution):")
counts = Counter(dois)
dist = Counter(counts.values())
for n_rows, n_papers in sorted(dist.items()):
    print(f"  {n_rows} row(s): {n_papers} paper(s)")
print(f"  max rows from a single paper: {max(counts.values())}")
print(f"  median rows/paper: {sorted(counts.values())[len(counts)//2]}")

print("\nIDENTITY/LINKAGE STATUS COUNTS (rows in final dataset):")
for col in ("material_identity_status", "processing_identity_status", "test_conditions_status", "stage3_observation_status"):
    c = Counter(r.get(col) for r in rows)
    print(f"  {col}: {dict(c)}")

print("\nCOMPOSITION COVERAGE (rows with at least one non-null core element):")
core_elements = ["C", "Mn", "Si", "Al", "Cr", "Ni", "Mo", "Nb", "Ti", "V", "Cu", "P", "S", "N"]
with_comp = sum(1 for r in rows if any(r.get(e) not in (None, "") for e in core_elements))
print(f"  rows with >=1 composition element: {with_comp} / {len(rows)}")
for e in core_elements:
    n = sum(1 for r in rows if r.get(e) not in (None, ""))
    if n:
        print(f"    {e}: {n}")

print("\nPROVENANCE VERIFIED COUNTS:")
for col in ("UTS_provenance_verified", "YS_provenance_verified", "elongation_provenance_verified"):
    c = Counter(r.get(col) for r in rows)
    print(f"  {col}: {dict(c)}")

print("\n" + "="*70)
print("EXCLUDED / NO-USABLE-TARGET REASON BREAKDOWN")
with (OUT_DIR / "cpp_dataset_v2_excluded_summary.csv").open() as f:
    excl_rows = list(csv.DictReader(f))
print(f"total excluded records: {len(excl_rows)}")
reason_counts = Counter(r.get("reason") for r in excl_rows)
for reason, n in reason_counts.most_common():
    print(f"  {n:5d}  {reason}")

source_counts = Counter(r.get("exclusion_source") for r in excl_rows)
print("\nexclusion_source breakdown:")
for source, n in source_counts.most_common():
    print(f"  {n:5d}  {source}")
