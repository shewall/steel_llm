"""
test_dataset_contract.py — plain-assert style tests (matching this project's
existing test_evidence_prep_v2.py / test_stage3_gate.py convention), synthetic
data only. No real CPP data, no LLM calls.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from literature_pipeline.modelling.dataset_contract import (  # noqa: E402
    REQUIRED_IDENTITY_COLUMNS,
    TARGET_COLUMNS,
    GROUP_COLUMN,
    ProvenanceColumnMap,
    make_and_predicate,
    make_status_in_predicate,
    make_target_has_value_predicate,
    make_target_provenance_predicate,
    validate_minimum_structure,
)

FAILURES = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not cond:
        FAILURES.append(label)


def make_minimal_df(n=4):
    return pd.DataFrame({
        "doi": [f"10.1/paper{i % 2}" for i in range(n)],
        "material_id": [f"m{i}" for i in range(n)],
        "condition_id": [f"c{i}" for i in range(n)],
        "observation_id": [f"o{i}" for i in range(n)],
        "UTS_MPa": [800.0, None, 900.0, None],
        "YS_MPa": [None, 500.0, None, None],
        "elongation_pct": [None, None, None, 12.0],
    })


# ─────────────────────────────────────────────────────────────────────────
# validate_minimum_structure — success path
# ─────────────────────────────────────────────────────────────────────────
df_ok = make_minimal_df()
report_ok = validate_minimum_structure(df_ok)
check("minimal valid df -> ok=True", report_ok.ok is True)
check("minimal valid df -> no missing required columns", report_ok.missing_required_columns == [])
check("minimal valid df -> no missing target columns", report_ok.missing_target_columns == [])
check("minimal valid df -> has_any_target_data True", report_ok.has_any_target_data is True)
check("minimal valid df -> no warnings", report_ok.warnings == [])

# ─────────────────────────────────────────────────────────────────────────
# missing required identity columns
# ─────────────────────────────────────────────────────────────────────────
df_missing_identity = df_ok.drop(columns=["condition_id"])
report_missing_identity = validate_minimum_structure(df_missing_identity)
check("dropping condition_id -> ok=False", report_missing_identity.ok is False)
check("dropping condition_id -> reported in missing_required_columns",
      report_missing_identity.missing_required_columns == ["condition_id"])

# ─────────────────────────────────────────────────────────────────────────
# all three targets required to exist as columns
# ─────────────────────────────────────────────────────────────────────────
df_missing_target_col = df_ok.drop(columns=["YS_MPa"])
report_missing_target_col = validate_minimum_structure(df_missing_target_col)
check("dropping YS_MPa column entirely -> ok=False", report_missing_target_col.ok is False)
check("dropping YS_MPa column entirely -> reported in missing_target_columns",
      report_missing_target_col.missing_target_columns == ["YS_MPa"])

# ─────────────────────────────────────────────────────────────────────────
# targets present as columns but entirely null everywhere -> has_any_target_data False
# ─────────────────────────────────────────────────────────────────────────
df_all_null_targets = df_ok.copy()
for c in TARGET_COLUMNS:
    df_all_null_targets[c] = None
report_all_null = validate_minimum_structure(df_all_null_targets)
check("all target columns present but entirely null -> ok=False", report_all_null.ok is False)
check("all target columns present but entirely null -> has_any_target_data False",
      report_all_null.has_any_target_data is False)
check("all target columns present but entirely null -> warning emitted",
      any("no non-null values" in w for w in report_all_null.warnings))

# a row missing YS/elongation must NOT disqualify overall has_any_target_data
# (this is exactly the per-row nullability requirement — targets independently nullable)
check("independently-nullable targets: UTS present in some rows is enough for has_any_target_data",
      report_ok.has_any_target_data is True)

# ─────────────────────────────────────────────────────────────────────────
# provenance map optional; when given, missing provenance columns detected
# ─────────────────────────────────────────────────────────────────────────
report_no_map = validate_minimum_structure(df_ok, provenance_map=None)
check("provenance_map=None -> ok still True (provenance optional)", report_no_map.ok is True)
check("provenance_map=None -> missing_provenance_columns empty", report_no_map.missing_provenance_columns == [])

pmap = ProvenanceColumnMap(
    material_identity_status_column="material_status",
    processing_identity_status_column="processing_status",
    target_provenance_verified_column={"UTS_MPa": "uts_provenance_ok", "YS_MPa": "ys_provenance_ok",
                                        "elongation_pct": "elong_provenance_ok"},
    stage3_observation_status_column="stage3_status",
)
report_with_map_missing_cols = validate_minimum_structure(df_ok, provenance_map=pmap)
check("provenance_map given but none of its columns exist -> ok=False",
      report_with_map_missing_cols.ok is False)
check("provenance_map given but none of its columns exist -> all 5 columns reported missing",
      sorted(report_with_map_missing_cols.missing_provenance_columns) ==
      sorted(["material_status", "processing_status", "uts_provenance_ok", "ys_provenance_ok",
              "elong_provenance_ok", "stage3_status"]))

df_with_provenance_cols = df_ok.copy()
df_with_provenance_cols["material_status"] = "confirmed"
df_with_provenance_cols["processing_status"] = "confirmed"
df_with_provenance_cols["uts_provenance_ok"] = True
df_with_provenance_cols["ys_provenance_ok"] = True
df_with_provenance_cols["elong_provenance_ok"] = True
df_with_provenance_cols["stage3_status"] = "valid_observation"
report_with_map_present = validate_minimum_structure(df_with_provenance_cols, provenance_map=pmap)
check("provenance_map given and all columns present -> ok=True", report_with_map_present.ok is True)

# stage2_version_column optional within the map itself
pmap_no_stage2 = ProvenanceColumnMap(
    material_identity_status_column="material_status",
    processing_identity_status_column="processing_status",
    target_provenance_verified_column={"UTS_MPa": "uts_provenance_ok"},
)
check("ProvenanceColumnMap.stage2_version_column defaults to None", pmap_no_stage2.stage2_version_column is None)
check("ProvenanceColumnMap.stage3_observation_status_column defaults to None",
      pmap_no_stage2.stage3_observation_status_column is None)

# ─────────────────────────────────────────────────────────────────────────
# predicate builders — missing-column-safe (return all-False, never raise)
# ─────────────────────────────────────────────────────────────────────────
pred_missing_status_col = make_status_in_predicate("nonexistent_col", {"confirmed"})
check("make_status_in_predicate on missing column returns all-False, does not raise",
      (pred_missing_status_col(df_ok) == False).all())  # noqa: E712

pred_missing_target = make_target_has_value_predicate("nonexistent_target")
check("make_target_has_value_predicate on missing column returns all-False, does not raise",
      (pred_missing_target(df_ok) == False).all())  # noqa: E712

pred_uts_has_value = make_target_has_value_predicate("UTS_MPa")
check("make_target_has_value_predicate(UTS_MPa) matches exactly the non-null UTS rows",
      list(pred_uts_has_value(df_ok)) == list(df_ok["UTS_MPa"].notna()))

pred_status = make_status_in_predicate("material_status", {"confirmed", "high_confidence"})
check("make_status_in_predicate matches allowed values only",
      pred_status(df_with_provenance_cols).all())
df_mixed_status = df_with_provenance_cols.copy()
df_mixed_status.loc[0, "material_status"] = "unconfirmed"
check("make_status_in_predicate excludes disallowed value",
      pred_status(df_mixed_status).tolist() == [False, True, True, True])

# target-scoped provenance predicate — the critical target-specificity check
pred_uts_only_provenance = make_target_provenance_predicate(
    pmap.target_provenance_verified_column, targets=["UTS_MPa"])
df_ys_provenance_missing = df_with_provenance_cols.copy()
df_ys_provenance_missing["ys_provenance_ok"] = False  # YS provenance NOT verified anywhere
check("target-scoped provenance predicate for UTS_MPa ignores YS/elongation provenance state",
      pred_uts_only_provenance(df_ys_provenance_missing).all())

pred_all_targets_provenance = make_target_provenance_predicate(pmap.target_provenance_verified_column)
check("un-scoped provenance predicate (all targets) DOES fail when YS provenance is false",
      not pred_all_targets_provenance(df_ys_provenance_missing).all())

# make_and_predicate combinator
combined = make_and_predicate(pred_status, pred_uts_only_provenance)
check("make_and_predicate combines multiple predicates with AND",
      combined(df_ys_provenance_missing).tolist() ==
      (pred_status(df_ys_provenance_missing) & pred_uts_only_provenance(df_ys_provenance_missing)).tolist())

check("GROUP_COLUMN is 'doi'", GROUP_COLUMN == "doi")
check("REQUIRED_IDENTITY_COLUMNS includes doi/material_id/condition_id/observation_id",
      set(REQUIRED_IDENTITY_COLUMNS) == {"doi", "material_id", "condition_id", "observation_id"})
check("TARGET_COLUMNS is exactly the three stable targets",
      TARGET_COLUMNS == ["UTS_MPa", "YS_MPa", "elongation_pct"])

print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
