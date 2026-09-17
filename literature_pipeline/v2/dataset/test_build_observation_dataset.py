"""
test_build_observation_dataset.py — offline tests for the Stage 3 -> flat
observation dataset builder (build_observation_dataset.py). No network/model
calls anywhere. Synthetic Stage 3 JSON fixtures mirror the real schema
(cross-checked against data/extracted_v2/{baseline6,final6}/ during design),
plus two structural regression checks (27/28) against those real, explicitly
non-final example directories.

Run: python literature_pipeline/v2/dataset/test_build_observation_dataset.py
"""

import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from literature_pipeline.v2.dataset import build_observation_dataset as bod
from literature_pipeline.modelling import dataset_contract as dc
from literature_pipeline.modelling import modelling_lib as ml

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not condition:
        FAILURES.append(label)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic fixture builders — mirror the real Stage 3 schema exactly
# ─────────────────────────────────────────────────────────────────────────

def elem(element, value, qualifier, range_low=None, range_high=None, reported_value=None):
    return {
        "element": element,
        "reported_value": reported_value if reported_value is not None else str(value if value is not None else ""),
        "reported_unit": "wt.%",
        "value": value,
        "qualifier": qualifier,
        "range_low": range_low,
        "range_high": range_high,
        "provenance": {"location": {"type": "table", "table_index": 0, "row_index": 0, "column": element}, "verified": True},
    }


def prop(canonical, value, row_key=None, test_status="default_implicit", replicate_info=None, section_context=None,
         source_column=None, reported_unit=None, unit_verification_status=None, elongation_type=None,
         property_scope=None, loading_mode=None):
    """source_column/reported_unit/unit_verification_status default to None,
    matching a property with no confirmed unit information -- for a genuine
    pre-unit-provenance-patch record (keys entirely ABSENT, not merely None),
    use prop_legacy() instead; map_targets()'s .get()-based reads must treat
    both the same way, which is exactly what test 10 checks."""
    return {
        "canonical_property": canonical,
        "value": value,
        "source_context": {
            "material_identifier": None, "processing_identifier": None,
            "specimen_identifier": None, "specimen_identifier_unique_state": False,
            "is_baseline_or_as_received": True,
            "test_condition": {"status": test_status, "value": None},
            "row_key": row_key, "table_caption": "Synthetic table", "section_context": section_context,
        },
        "replicate_info": replicate_info or {"is_aggregate": False, "n_replicates_reported": None, "raw_replicate_values": None},
        "source_column": source_column, "reported_unit": reported_unit,
        "unit_verification_status": unit_verification_status,
        "elongation_type": elongation_type,
        "property_scope": property_scope, "loading_mode": loading_mode,
    }


def prop_legacy(canonical, value, row_key=None, test_status="default_implicit"):
    """A property dict shaped exactly like real pre-unit-provenance-patch
    Stage 3 output (e.g. the existing baseline6/final6 files): no
    source_column/reported_unit/unit_verification_status keys AT ALL, not
    merely None-valued. Used to test genuine backward compatibility."""
    return {
        "canonical_property": canonical,
        "value": value,
        "source_context": {
            "material_identifier": None, "processing_identifier": None,
            "specimen_identifier": None, "specimen_identifier_unique_state": False,
            "is_baseline_or_as_received": True,
            "test_condition": {"status": test_status, "value": None},
            "row_key": row_key, "table_caption": "Synthetic table", "section_context": None,
        },
        "replicate_info": {"is_aggregate": False, "n_replicates_reported": None, "raw_replicate_values": None},
    }


def make_observation(obs_id, material_id, group_ref, properties, material_identity="confirmed",
                      processing_identity="confirmed_explicit", test_conditions_status="default_implicit",
                      status="valid_observation"):
    return {
        "observation_status": status,
        "linkage": {"material_identity": material_identity, "processing_identity": processing_identity,
                    "test_conditions_status": test_conditions_status},
        "property_completeness": {
            "properties_present": sorted(properties.keys()),
            "completeness_tier": {1: "single_property", 2: "two_properties", 3: "three_properties"}.get(
                len(properties), f"{len(properties)}_properties"),
        },
        "properties": properties,
        "observation_id": obs_id, "material_id": material_id, "stage2_evidence_group_ref": group_ref,
    }


def make_condition(condition_id, processing_key, observations):
    return {"condition_id": condition_id, "processing_key": processing_key, "observations": observations}


def make_material(material_id, grade_label, elements, conditions, composition_status="reported"):
    return {"material_id": material_id, "reported_grade_label": grade_label,
            "composition": {"status": composition_status, "elements": elements},
            "processing_conditions": conditions}


def make_stage3_record(doi, materials, unresolved=None, rejected=None, failures=None, code_version="test_v0"):
    return {"doi": doi, "stage3_code_version": code_version, "stage3_prompt_version": "test",
            "materials": materials, "unresolved_candidates": unresolved or [],
            "rejected_candidates": rejected or [], "failures": failures or []}


def write_record(tmpdir, filename, record):
    path = Path(tmpdir) / filename
    path.write_text(json.dumps(record))
    return path


# ─────────────────────────────────────────────────────────────────────────
# 1. one paper / one material / one valid observation
# ─────────────────────────────────────────────────────────────────────────
rec1 = make_stage3_record(
    "10.1/paper1",
    [make_material("M1", "DP780", [elem("C", 0.08, "exact"), elem("Mn", 1.5, "exact")],
                    [make_condition("M1-P1", "IA700", [
                        make_observation("M1-P1-O1", "M1", "G1", {"UTS": prop("UTS", 980.0, row_key=[0, 0])}),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    path = write_record(tmp, "p1.json", rec1)
    result = bod.build_dataset([path])
check("1. one paper/one material/one observation -> exactly one row", len(result.dataset) == 1)
check("1. row has correct identity columns",
      result.dataset.iloc[0][["doi", "material_id", "condition_id", "observation_id"]].tolist()
      == ["10.1/paper1", "M1", "M1-P1", "M1-P1-O1"])

# ─────────────────────────────────────────────────────────────────────────
# 2. multiple materials
# ─────────────────────────────────────────────────────────────────────────
rec2 = make_stage3_record(
    "10.1/paper2",
    [
        make_material("M1", "DP780", [elem("C", 0.08, "exact")],
                       [make_condition("M1-P1", "IA700", [make_observation("M1-P1-O1", "M1", "G1", {"UTS": prop("UTS", 980.0)})])]),
        make_material("M2", "DP980", [elem("C", 0.15, "exact")],
                       [make_condition("M2-P1", "IA750", [make_observation("M2-P1-O1", "M2", "G1", {"UTS": prop("UTS", 1050.0)})])]),
    ],
)
with tempfile.TemporaryDirectory() as tmp:
    result2 = bod.build_dataset([write_record(tmp, "p2.json", rec2)])
check("2. multiple materials -> two rows, distinct material_id",
      len(result2.dataset) == 2 and set(result2.dataset["material_id"]) == {"M1", "M2"})

# ─────────────────────────────────────────────────────────────────────────
# 3. multiple processing conditions (one material)
# ─────────────────────────────────────────────────────────────────────────
rec3 = make_stage3_record(
    "10.1/paper3",
    [make_material("M1", "DP780", [elem("C", 0.08, "exact")],
                    [make_condition("M1-P1", "IA700", [make_observation("M1-P1-O1", "M1", "G1", {"UTS": prop("UTS", 980.0)})]),
                     make_condition("M1-P2", "IA750", [make_observation("M1-P2-O1", "M1", "G2", {"UTS": prop("UTS", 1010.0)})])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result3 = bod.build_dataset([write_record(tmp, "p3.json", rec3)])
check("3. multiple processing conditions -> two rows, distinct condition_id",
      len(result3.dataset) == 2 and set(result3.dataset["condition_id"]) == {"M1-P1", "M1-P2"})

# ─────────────────────────────────────────────────────────────────────────
# 4. multiple observations under one condition
# ─────────────────────────────────────────────────────────────────────────
rec4 = make_stage3_record(
    "10.1/paper4",
    [make_material("M1", "DP780", [elem("C", 0.08, "exact")],
                    [make_condition("M1-P1", "IA700", [
                        make_observation("M1-P1-O1", "M1", "G1", {"UTS": prop("UTS", 980.0)}),
                        make_observation("M1-P1-O2", "M1", "G2", {"YS": prop("YS", 720.0)}),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result4 = bod.build_dataset([write_record(tmp, "p4.json", rec4)])
check("4. multiple observations under one condition -> two rows, same condition_id",
      len(result4.dataset) == 2 and result4.dataset["condition_id"].nunique() == 1)

# ─────────────────────────────────────────────────────────────────────────
# 5/6/7/8. target presence combinations
# ─────────────────────────────────────────────────────────────────────────
rec_targets = make_stage3_record(
    "10.1/paper5",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [
                        make_observation("O_uts", "M1", "G1", {"UTS": prop("UTS", 900.0)}),
                        make_observation("O_ys", "M1", "G2", {"YS": prop("YS", 700.0)}),
                        make_observation("O_el", "M1", "G3", {"elongation": prop("elongation", 20.0)}),
                        make_observation("O_multi", "M1", "G4", {"UTS": prop("UTS", 1000.0), "YS": prop("YS", 800.0),
                                                                   "elongation": prop("elongation", 15.0)}),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result_t = bod.build_dataset([write_record(tmp, "p5.json", rec_targets)])
row = lambda oid: result_t.dataset[result_t.dataset["observation_id"] == oid].iloc[0]
check("5. UTS-only row: UTS_MPa set, YS/elongation NaN",
      row("O_uts")["UTS_MPa"] == 900.0 and pd.isna(row("O_uts")["YS_MPa"]) and pd.isna(row("O_uts")["elongation_pct"]))
check("6. YS-only row: YS_MPa set, UTS/elongation NaN",
      row("O_ys")["YS_MPa"] == 700.0 and pd.isna(row("O_ys")["UTS_MPa"]) and pd.isna(row("O_ys")["elongation_pct"]))
check("7. elongation-only row: elongation_pct set, UTS/YS NaN",
      row("O_el")["elongation_pct"] == 20.0 and pd.isna(row("O_el")["UTS_MPa"]) and pd.isna(row("O_el")["YS_MPa"]))
check("8. multi-target row: all three populated",
      row("O_multi")["UTS_MPa"] == 1000.0 and row("O_multi")["YS_MPa"] == 800.0 and row("O_multi")["elongation_pct"] == 15.0)

# ─────────────────────────────────────────────────────────────────────────
# 9/10/11/12. composition qualifier handling
# ─────────────────────────────────────────────────────────────────────────
rec_comp = make_stage3_record(
    "10.1/paper6",
    [make_material("M1", "G", [
        # Note: "Fe" is deliberately not part of dataset_contract.CORE_COMPOSITION_ELEMENTS
        # (confirmed: C, Mn, Si, Al, Cr, Ni, Mo, Nb, Ti, V, Cu, P, S, N), so a
        # "balance" qualifier is exercised here on Cr instead -- any core
        # element can legitimately carry a "balance" qualifier in principle;
        # this is a unit test of the qualifier pass-through, not a claim
        # about what real steel papers actually report as balance.
        elem("Cr", None, "balance"),
        elem("C", None, "range", range_low=0.08, range_high=0.12),
        elem("Mn", 0.01, "less_than"),
        elem("Ni", 2.0, "greater_than"),
        # note: Al deliberately absent from the elements list entirely
    ], [make_condition("M1-P1", None, [make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 900.0)})])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result_c = bod.build_dataset([write_record(tmp, "p6.json", rec_comp)])
crow = result_c.dataset.iloc[0]
check("9. element never reported at all (Al) -> NaN, qualifier None",
      pd.isna(crow["Al"]) and crow["Al_qualifier"] is None)
check("10. balance -> numeric NaN, qualifier 'balance' preserved",
      pd.isna(crow["Cr"]) and crow["Cr_qualifier"] == "balance")
check("11. range qualifier -> numeric NaN (no midpoint invented), qualifier 'range' preserved",
      pd.isna(crow["C"]) and crow["C_qualifier"] == "range")
check("12a. less_than qualifier -> Stage 3's own threshold value passed through unchanged, qualifier preserved",
      crow["Mn"] == 0.01 and crow["Mn_qualifier"] == "less_than")
check("12b. greater_than qualifier -> Stage 3's own threshold value passed through unchanged, qualifier preserved",
      crow["Ni"] == 2.0 and crow["Ni_qualifier"] == "greater_than")

# ─────────────────────────────────────────────────────────────────────────
# 13/14. orphan / rejected excluded
# ─────────────────────────────────────────────────────────────────────────
rec_excl = make_stage3_record(
    "10.1/paper7",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 900.0)})])])],
    unresolved=[{"material_id": "M1", "group_id": "G2", "observation_status": "orphan_property_evidence",
                 "reason": "material_identity=assumed_single_material",
                 "properties": [{"property": "YS", "value": 700.0}]}],
    rejected=[{"material_id": "M1", "group_id": "G3", "property": "elongation",
               "reason": "unsupported_numerical_token"}],
)
with tempfile.TemporaryDirectory() as tmp:
    result_e = bod.build_dataset([write_record(tmp, "p7.json", rec_excl)])
check("13. orphan_property_evidence never becomes a modelling row",
      len(result_e.dataset) == 1 and "O1" == result_e.dataset.iloc[0]["observation_id"])
check("13b. orphan preserved in excluded_records with its reason",
      any(r["exclusion_source"] == "orphan_property_evidence" and r["reason"] == "material_identity=assumed_single_material"
          for r in result_e.excluded_records))
check("14. rejected_candidate never becomes a modelling row and is preserved with reason",
      any(r["exclusion_source"] == "rejected_candidate" and r["reason"] == "unsupported_numerical_token"
          for r in result_e.excluded_records))

# ─────────────────────────────────────────────────────────────────────────
# 15/16. duplicate identity -> hard failure (tested directly against
# audit_duplicates for precise control over the trigger)
# ─────────────────────────────────────────────────────────────────────────
dup_df = pd.DataFrame([
    {**{c: None for c in bod.ALL_COLUMNS}, "doi": "10.1/x", "material_id": "M1", "condition_id": "M1-P1",
     "observation_id": "M1-P1-O1", "UTS_MPa": 900.0, "YS_MPa": float("nan"), "elongation_pct": float("nan")},
    {**{c: None for c in bod.ALL_COLUMNS}, "doi": "10.1/x", "material_id": "M1", "condition_id": "M1-P1",
     "observation_id": "M1-P1-O1", "UTS_MPa": 950.0, "YS_MPa": float("nan"), "elongation_pct": float("nan")},
], columns=bod.ALL_COLUMNS)
try:
    bod.audit_duplicates(dup_df, [])
    check("15/16. duplicate observation_id and duplicate identity tuple both raise DuplicateIdentityError", False)
except bod.DuplicateIdentityError as e:
    msg = str(e)
    check("15. duplicate observation_id detected in the raised error", "duplicate observation_id" in msg)
    check("16. duplicate (doi, material_id, condition_id, observation_id) tuple detected in the raised error",
          "duplicate (doi, material_id, condition_id, observation_id)" in msg)

# ─────────────────────────────────────────────────────────────────────────
# 17. duplicate source evidence -> warning only, never deletion
# ─────────────────────────────────────────────────────────────────────────
shared_loc_audit = [
    {"observation_id": "O1", "doi": "10.1/y", "properties": {"UTS": {"source_context": {"row_key": [0, 0]}}}},
    {"observation_id": "O2", "doi": "10.1/y", "properties": {"UTS": {"source_context": {"row_key": [0, 0]}}}},
]
clean_df = pd.DataFrame([
    {**{c: None for c in bod.ALL_COLUMNS}, "doi": "10.1/y", "material_id": "M1", "condition_id": "M1-P1",
     "observation_id": "O1", "UTS_MPa": 900.0, "YS_MPa": float("nan"), "elongation_pct": float("nan")},
    {**{c: None for c in bod.ALL_COLUMNS}, "doi": "10.1/y", "material_id": "M1", "condition_id": "M1-P1",
     "observation_id": "O2", "UTS_MPa": 950.0, "YS_MPa": float("nan"), "elongation_pct": float("nan")},
], columns=bod.ALL_COLUMNS)
dup_report = bod.audit_duplicates(clean_df, shared_loc_audit)
check("17a. shared source location does NOT raise (rows preserved)", len(clean_df) == 2)
check("17b. shared source location IS recorded as a warning in the duplicate report",
      len(dup_report["shared_source_location"]) == 1 and set(dup_report["shared_source_location"][0]["observation_ids"]) == {"O1", "O2"})

# ─────────────────────────────────────────────────────────────────────────
# 18/19/20. dataset_contract + modelling_lib compatibility
# ─────────────────────────────────────────────────────────────────────────
rec_prov = make_stage3_record(
    "10.1/paperA",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [
                        make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 900.0)},
                                          material_identity="confirmed", processing_identity="confirmed_explicit"),
                    ])])],
)
rec_prov_weak = make_stage3_record(
    "10.1/paperB",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [
                        make_observation("O2", "M1", "G1", {"UTS": prop("UTS", 950.0)},
                                          material_identity="confirmed", processing_identity="confirmed_baseline"),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result_p = bod.build_dataset([write_record(tmp, "pA.json", rec_prov), write_record(tmp, "pB.json", rec_prov_weak)])

report = dc.validate_minimum_structure(result_p.dataset)
check("19. builder output passes dataset_contract.validate_minimum_structure",
      report.ok is True, detail=str(report))

pmap = dc.ProvenanceColumnMap(
    material_identity_status_column="material_identity_status",
    processing_identity_status_column="processing_identity_status",
    target_provenance_verified_column={"UTS_MPa": "UTS_provenance_verified", "YS_MPa": "YS_provenance_verified",
                                        "elongation_pct": "elongation_provenance_verified"},
    stage3_observation_status_column="stage3_observation_status",
    stage2_version_column=None,
)
check("18. all ProvenanceColumnMap-referenced columns exist in builder output",
      all(col in result_p.dataset.columns for col in
          [pmap.material_identity_status_column, pmap.processing_identity_status_column,
           pmap.stage3_observation_status_column, *pmap.target_provenance_verified_column.values()]))

try:
    tiers = ml.build_target_quality_tiers(result_p.dataset, "UTS_MPa", pmap,
                                           material_confirmed_values={"confirmed"},
                                           processing_confirmed_values={"confirmed_explicit", "confirmed_baseline"})
    tier_frames = ml.apply_quality_tiers(result_p.dataset, tiers)
    check("20. build_target_quality_tiers/apply_quality_tiers run against builder output without error",
          set(tier_frames["UTS_MPa_usable"]["doi"]) == {"10.1/paperA", "10.1/paperB"})
except Exception as e:
    check("20. build_target_quality_tiers/apply_quality_tiers run against builder output without error", False, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────
# 21. synthetic replicate case — preserved, never aggregated
# ─────────────────────────────────────────────────────────────────────────
rec_rep = make_stage3_record(
    "10.1/paper8",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [
                        make_observation("O1", "M1", "G1", {
                            "UTS": prop("UTS", None, replicate_info={"is_aggregate": False, "n_replicates_reported": 2,
                                                                       "raw_replicate_values": [1200.0, 1210.0]}),
                            "YS": prop("YS", 800.0),
                        }),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result_r = bod.build_dataset([write_record(tmp, "p8.json", rec_rep)])
rrow = result_r.dataset.iloc[0]
check("21a. row is included (YS gives a usable target even though UTS is replicate-unresolved)", len(result_r.dataset) == 1)
check("21b. UTS_MPa stays NaN (never averaged/median/first-value selected)", pd.isna(rrow["UTS_MPa"]))
check("21c. YS_MPa is the real value, unaffected", rrow["YS_MPa"] == 800.0)
aud = next(a for a in result_r.audit_records if a["observation_id"] == "O1")
check("21d. raw_replicate_values preserved verbatim in the audit sidecar",
      aud["properties"]["UTS"]["replicate_info"]["raw_replicate_values"] == [1200.0, 1210.0])

# ─────────────────────────────────────────────────────────────────────────
# 22. observation with ONLY unresolved replicate target(s) -> excluded from
# CSV, preserved in excluded output with the exact required reason string
# ─────────────────────────────────────────────────────────────────────────
rec_rep_only = make_stage3_record(
    "10.1/paper9",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [
                        make_observation("O1", "M1", "G1", {
                            "UTS": prop("UTS", None, replicate_info={"is_aggregate": False, "n_replicates_reported": 2,
                                                                       "raw_replicate_values": [1200.0, 1210.0]}),
                        }),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result_ro = bod.build_dataset([write_record(tmp, "p9.json", rec_rep_only)])
check("22a. observation with only an unresolved-replicate target is excluded from the modelling CSV",
      len(result_ro.dataset) == 0)
check("22b. excluded with the exact required reason 'replicate_target_unresolved'",
      any(r["reason"] == "replicate_target_unresolved" and r["observation_id"] == "O1"
          for r in result_ro.excluded_records))

# ─────────────────────────────────────────────────────────────────────────
# 23. empty Stage 3 record -> zero rows, no crash
# ─────────────────────────────────────────────────────────────────────────
rec_empty = make_stage3_record("10.1/paper10", [])
with tempfile.TemporaryDirectory() as tmp:
    result_empty = bod.build_dataset([write_record(tmp, "p10.json", rec_empty)])
check("23. empty Stage 3 record (zero materials) -> zero rows, no crash", len(result_empty.dataset) == 0)
check("23b. zero-row dataset still has the full expected column set",
      list(result_empty.dataset.columns) == bod.ALL_COLUMNS)

# ─────────────────────────────────────────────────────────────────────────
# 24. processing_key = None handled cleanly
# ─────────────────────────────────────────────────────────────────────────
rec_none_key = make_stage3_record(
    "10.1/paper11",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 900.0)})])])],
)
with tempfile.TemporaryDirectory() as tmp:
    result_nk = bod.build_dataset([write_record(tmp, "p11.json", rec_none_key)])
check("24. processing_key=None is represented as null, not the string 'None', and does not crash",
      result_nk.dataset.iloc[0]["processing_key"] is None)

# ─────────────────────────────────────────────────────────────────────────
# 25. multi-paper build
# ─────────────────────────────────────────────────────────────────────────
rec_p1 = make_stage3_record("10.1/multiA", [make_material("M1", "G", [elem("C", 0.1, "exact")],
                             [make_condition("M1-P1", None, [make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 900.0)})])])])
rec_p2 = make_stage3_record("10.1/multiB", [make_material("M1", "G", [elem("C", 0.1, "exact")],
                             [make_condition("M1-P1", None, [make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 950.0)})])])])
with tempfile.TemporaryDirectory() as tmp:
    result_multi = bod.build_dataset([write_record(tmp, "mA.json", rec_p1), write_record(tmp, "mB.json", rec_p2)])
check("25. multi-paper build produces rows from both papers with distinct DOIs",
      set(result_multi.dataset["doi"]) == {"10.1/multiA", "10.1/multiB"} and len(result_multi.dataset) == 2)
check("25b. identical observation_id string across two different DOIs does not falsely collide "
      "(doi is part of the identity, not just observation_id)",
      len(result_multi.dataset) == 2)

# ─────────────────────────────────────────────────────────────────────────
# 26 (updated). Absence of unit information is explicitly, correctly
# reported in the new per-target QC unit-coverage structure (the old
# blanket target_unit_verification_status string was replaced by real
# per-target counts — see the unit-provenance test block below).
# ─────────────────────────────────────────────────────────────────────────
check("26. QC report's target_unit_coverage correctly reports the multi-paper build's two "
      "unit-less (prop() defaults) UTS observations as unit-not-available",
      result_multi.qc_report["target_unit_coverage"]["UTS_MPa"]["n_target_values"] == 2
      and result_multi.qc_report["target_unit_coverage"]["UTS_MPa"]["n_unit_not_available"] == 2
      and result_multi.qc_report["target_unit_coverage"]["UTS_MPa"]["n_expected_unit_verified"] == 0)

# ─────────────────────────────────────────────────────────────────────────
# PINNED SCHEMA TEST — independent, hard-coded expected column list, not
# derived from the live ALL_COLUMNS constant. Closes the gap identified
# earlier: a test that compares a DataFrame's columns against ALL_COLUMNS
# itself can never catch an accidental change to ALL_COLUMNS; this one can.
# ─────────────────────────────────────────────────────────────────────────
_PINNED_EXPECTED_SCHEMA = [
    "doi", "material_id", "condition_id", "observation_id",
    "reported_grade_label", "stage2_evidence_group_ref", "processing_key",
    "UTS_MPa", "YS_MPa", "elongation_pct",
    "C", "C_qualifier", "Mn", "Mn_qualifier", "Si", "Si_qualifier", "Al", "Al_qualifier",
    "Cr", "Cr_qualifier", "Ni", "Ni_qualifier", "Mo", "Mo_qualifier", "Nb", "Nb_qualifier",
    "Ti", "Ti_qualifier", "V", "V_qualifier", "Cu", "Cu_qualifier", "P", "P_qualifier",
    "S", "S_qualifier", "N", "N_qualifier",
    "material_identity_status", "processing_identity_status", "test_conditions_status",
    "stage3_observation_status",
    "UTS_provenance_verified", "YS_provenance_verified", "elongation_provenance_verified",
    "UTS_reported_unit", "YS_reported_unit", "elongation_reported_unit",
    "UTS_unit_verified", "YS_unit_verified", "elongation_unit_verified",
    "elongation_type",
    "stage3_code_version",
]
check("u16. explicit pinned 53-column expected schema matches the actual builder output schema exactly",
      list(bod.ALL_COLUMNS) == _PINNED_EXPECTED_SCHEMA,
      detail=f"len(ALL_COLUMNS)={len(bod.ALL_COLUMNS)}, len(pinned)={len(_PINNED_EXPECTED_SCHEMA)}")
check("u16b. pinned schema also matches an actual built dataset's real columns",
      list(result_multi.dataset.columns) == _PINNED_EXPECTED_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────
# Unit-provenance builder tests (Stage 3 v0.4 patch consumption)
# ─────────────────────────────────────────────────────────────────────────

def _is_null(v):
    return v is None or (isinstance(v, float) and pd.isna(v))


def _one_row_build(properties, tmp_name="unit_test.json", doi="10.1/unit-test"):
    rec = make_stage3_record(doi, [make_material("M1", "G", [elem("C", 0.1, "exact")],
                              [make_condition("M1-P1", None, [make_observation("O1", "M1", "G1", properties)])])])
    with tempfile.TemporaryDirectory() as tmp:
        result = bod.build_dataset([write_record(tmp, tmp_name, rec)])
    return result


_r_u1 = _one_row_build({"UTS": prop("UTS", 980.0, source_column="UTS (MPa)", reported_unit="MPa",
                                     unit_verification_status="verified_from_table_header")})
_row_u1 = _r_u1.dataset.iloc[0]
check("u1. UTS MPa + verified_from_table_header -> UTS_reported_unit='MPa', UTS_unit_verified=True",
      _row_u1["UTS_reported_unit"] == "MPa" and _row_u1["UTS_unit_verified"] == True)

_r_u2 = _one_row_build({"UTS": prop("UTS", 980.0, reported_unit="MPa", unit_verification_status="verified_from_local_text")})
_row_u2 = _r_u2.dataset.iloc[0]
check("u2. UTS MPa + verified_from_local_text -> UTS_unit_verified=True",
      _row_u2["UTS_reported_unit"] == "MPa" and _row_u2["UTS_unit_verified"] == True)

_r_u3 = _one_row_build({"UTS": prop("UTS", 1.2, source_column="UTS (GPa)", reported_unit="GPa",
                                     unit_verification_status="verified_from_table_header")})
_row_u3 = _r_u3.dataset.iloc[0]
check("u3. UTS GPa (independently verified, but NOT the expected unit) -> UTS_reported_unit='GPa', "
      "UTS_unit_verified=False, AND the numeric UTS_MPa value is unchanged/uncoverted (still 1.2, no conversion)",
      _row_u3["UTS_reported_unit"] == "GPa" and _row_u3["UTS_unit_verified"] == False and _row_u3["UTS_MPa"] == 1.2)

_r_u4 = _one_row_build({"UTS": prop("UTS", 980.0)})
_row_u4 = _r_u4.dataset.iloc[0]
check("u4. UTS unit missing entirely -> UTS_reported_unit is null, UTS_unit_verified=False",
      _is_null(_row_u4["UTS_reported_unit"]) and _row_u4["UTS_unit_verified"] == False)

_r_u5 = _one_row_build({"UTS": prop("UTS", 980.0, reported_unit=None, unit_verification_status="reported_by_model_unverified")})
_row_u5 = _r_u5.dataset.iloc[0]
check("u5. UTS status=reported_by_model_unverified -> UTS_unit_verified=False",
      _row_u5["UTS_unit_verified"] == False)

_r_u6 = _one_row_build({"YS": prop("YS", 720.0, reported_unit="MPa", unit_verification_status="verified_from_table_header")})
_row_u6 = _r_u6.dataset.iloc[0]
check("u6. YS MPa verified -> YS_unit_verified=True", _row_u6["YS_unit_verified"] == True)

_r_u7 = _one_row_build({"elongation": prop("elongation", 22.0, reported_unit="%", unit_verification_status="verified_from_table_header")})
_row_u7 = _r_u7.dataset.iloc[0]
check("u7. elongation % verified -> elongation_unit_verified=True", _row_u7["elongation_unit_verified"] == True)

_r_u8 = _one_row_build({"elongation": prop("elongation", 22.0)})
_row_u8 = _r_u8.dataset.iloc[0]
check("u8. elongation unit missing/unverified -> elongation_unit_verified=False",
      _row_u8["elongation_unit_verified"] == False)

_r_u9 = _one_row_build({"UTS": prop("UTS", 980.0)})
_row_u9 = _r_u9.dataset.iloc[0]
check("u9. UTS_provenance_verified=True while UTS_unit_verified=False is a legitimate, independent state",
      _row_u9["UTS_provenance_verified"] == True and _row_u9["UTS_unit_verified"] == False)

_r_u10 = _one_row_build({"UTS": prop_legacy("UTS", 980.0)})
_row_u10 = _r_u10.dataset.iloc[0]
check("u10. legacy Stage 3 property (unit keys entirely absent, not just None) reads safely without crashing: "
      "UTS_MPa=980.0 preserved, UTS_reported_unit null, UTS_unit_verified=False, no unit inferred",
      _row_u10["UTS_MPa"] == 980.0 and _is_null(_row_u10["UTS_reported_unit"]) and _row_u10["UTS_unit_verified"] == False)

_r_u11 = _one_row_build({"UTS": prop("UTS", 980.0, source_column="UTS (MPa)", reported_unit="MPa",
                                      unit_verification_status="verified_from_table_header")})
_aud_u11 = _r_u11.audit_records[0]
check("u11. audit sidecar preserves source_column/reported_unit/unit_verification_status verbatim, unmodified",
      _aud_u11["properties"]["UTS"]["source_column"] == "UTS (MPa)"
      and _aud_u11["properties"]["UTS"]["reported_unit"] == "MPa"
      and _aud_u11["properties"]["UTS"]["unit_verification_status"] == "verified_from_table_header")

_rec_u12 = make_stage3_record(
    "10.1/unit-qc",
    [make_material("M1", "G", [elem("C", 0.1, "exact")],
                    [make_condition("M1-P1", None, [
                        make_observation("O1", "M1", "G1", {"UTS": prop("UTS", 900.0, reported_unit="MPa",
                                          unit_verification_status="verified_from_table_header")}),
                        make_observation("O2", "M1", "G2", {"UTS": prop("UTS", 1.0, reported_unit="GPa",
                                          unit_verification_status="verified_from_table_header")}),
                        make_observation("O3", "M1", "G3", {"UTS": prop("UTS", 950.0)}),
                    ])])],
)
with tempfile.TemporaryDirectory() as tmp:
    _result_u12 = bod.build_dataset([write_record(tmp, "u12.json", _rec_u12)])
_cov_u12 = _result_u12.qc_report["target_unit_coverage"]["UTS_MPa"]
check("u12. QC target_unit_coverage counts are correct across a 3-observation mix "
      "(1 verified MPa, 1 verified-but-other-unit GPa, 1 no-unit)",
      _cov_u12["n_target_values"] == 3 and _cov_u12["n_expected_unit_verified"] == 1
      and _cov_u12["n_verified_other_unit"] == 1 and _cov_u12["n_unit_not_available"] == 1
      and _cov_u12["n_unit_unverified"] == 0)

_r_u13a = _one_row_build({"UTS": prop("UTS", 1234.5)}, tmp_name="u13a.json", doi="10.1/u13a")
_r_u13b = _one_row_build({"UTS": prop("UTS", 1234.5, reported_unit="MPa", unit_verification_status="verified_from_table_header")},
                          tmp_name="u13b.json", doi="10.1/u13b")
check("u13. numeric target value is byte-identical regardless of unit metadata presence "
      "(this patch never touches value extraction)",
      _r_u13a.dataset.iloc[0]["UTS_MPa"] == _r_u13b.dataset.iloc[0]["UTS_MPa"] == 1234.5)

# ─────────────────────────────────────────────────────────────────────────
# write_dataset — writes to a temp dir only, never data/datasets_v2/ in tests
# ─────────────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    paths = bod.write_dataset(result_multi, out_dir=Path(tmp))
    check("bonus: write_dataset produces all five approved output files",
          all(p.exists() for p in paths.values()))
    check("bonus: write_dataset never touches data/datasets_v2/ during this test",
          all(str(Path(tmp)) in str(p) for p in paths.values()))

# ─────────────────────────────────────────────────────────────────────────
# 27/28. structural regression checks against the real, non-final six-paper
# directories. These are STRUCTURAL/REGRESSION checks only — they confirm
# the builder reproduces a known, already-verified count from real (but
# explicitly non-final, pre-full-corpus) Stage 3 output. They are NOT a
# claim about final scientific dataset content.
# ─────────────────────────────────────────────────────────────────────────
final6_files = sorted(f for f in glob.glob("data/extracted_v2/final6/*.json") if "_run" not in f)
baseline6_files = sorted(f for f in glob.glob("data/extracted_v2/baseline6/*.json") if "_run" not in f)

if final6_files:
    result_final6 = bod.build_dataset(final6_files)
    check("27. STRUCTURAL REGRESSION ONLY: final6 (real, non-final Stage 3 output) produces ZERO modelling rows",
          len(result_final6.dataset) == 0,
          detail=f"got {len(result_final6.dataset)} rows")
else:
    check("27. final6 files present for regression check", False, detail="no files found — skipped")

if baseline6_files:
    result_baseline6 = bod.build_dataset(baseline6_files)
    check("28. STRUCTURAL REGRESSION ONLY: baseline6 (real, non-final, PRE-SAFETY-FIX Stage 3 output) "
          "produces exactly ONE modelling row",
          len(result_baseline6.dataset) == 1,
          detail=f"got {len(result_baseline6.dataset)} rows")
else:
    check("28. baseline6 files present for regression check", False, detail="no files found — skipped")

# u14/u15: these real files pre-date the unit-provenance patch, so this also
# doubles as a live backward-compatibility check (no new unit keys present
# at all -- not merely None -- in every property record in both files).
if final6_files:
    check("u14. final6 STILL produces zero modelling rows after the unit-provenance patch "
          "(structural regression only, not a scientific result)",
          len(result_final6.dataset) == 0)
if baseline6_files:
    check("u15. baseline6 STILL produces exactly one modelling row after the unit-provenance patch "
          "(structural regression only, not a scientific result), and it reads safely with no unit keys present "
          "(UTS_reported_unit/YS_reported_unit null, *_unit_verified False, no crash, no inferred unit)",
          len(result_baseline6.dataset) == 1
          and _is_null(result_baseline6.dataset.iloc[0]["UTS_reported_unit"])
          and _is_null(result_baseline6.dataset.iloc[0]["YS_reported_unit"])
          and result_baseline6.dataset.iloc[0]["UTS_unit_verified"] == False
          and result_baseline6.dataset.iloc[0]["YS_unit_verified"] == False)

# ─────────────────────────────────────────────────────────────────────────
# Elongation semantics: single-type resolves normally with elongation_type
# preserved; multi-type conflict is left unresolved (never averaged, never
# silently picks one), distinct from a true unresolved-replicate reason.
# ─────────────────────────────────────────────────────────────────────────

# single uniform-elongation value -> elongation_pct populated, elongation_type preserved
obs_single_uniform = make_observation(
    "O1", "M1", "G1",
    {"UTS": prop("UTS", 830.0, source_column="UTS [MPa]", reported_unit="MPa", unit_verification_status="verified_from_table_header"),
     "elongation_uniform": prop("elongation", 8.8, elongation_type="uniform",
                                 source_column="UE [%]", reported_unit="%", unit_verification_status="verified_from_table_header")},
)
tr_single = bod.map_targets(obs_single_uniform)
check("elongation: single uniform value -> elongation_pct populated",
      tr_single.values["elongation_pct"] == 8.8, detail=str(tr_single.values))
check("elongation: elongation_type preserved as 'uniform'",
      tr_single.elongation_type == "uniform")
check("elongation: no conflict flagged for a single type",
      tr_single.elongation_type_conflict is False)
check("elongation: provenance_verified True for a resolved single type",
      tr_single.provenance_verified["elongation_provenance_verified"] is True)

# both uniform AND total present (the real DP1/DP2 shape) -> conflict,
# elongation_pct left unresolved, never averaged/silently picked, but
# provenance_verified stays True (real, verified evidence exists)
obs_conflict = make_observation(
    "O2", "M1", "G1",
    {"elongation_uniform": prop("elongation", 8.8, elongation_type="uniform"),
     "elongation_total": prop("elongation", 14.4, elongation_type="total")},
)
tr_conflict = bod.map_targets(obs_conflict)
check("elongation conflict: elongation_pct left unresolved (NaN), never averaged (not 11.6) or silently picked (not 8.8 or 14.4)",
      _is_null(tr_conflict.values["elongation_pct"]), detail=str(tr_conflict.values))
check("elongation conflict: elongation_type_conflict flagged True",
      tr_conflict.elongation_type_conflict is True)
check("elongation conflict: elongation_type itself is None (never guessed which one)",
      tr_conflict.elongation_type is None)
check("elongation conflict: provenance_verified stays True (real evidence exists, just not resolvable to one value)",
      tr_conflict.provenance_verified["elongation_provenance_verified"] is True)

# exclusion-reason check: an observation whose ONLY property is a
# conflicting elongation gets a DISTINCT reason from a true replicate case
material_conflict = make_material("M1", "TestGrade", [], [make_condition("C1", "baseline", [obs_conflict])])
record_conflict = make_stage3_record("10.9999/elongation-conflict", [material_conflict])
with tempfile.TemporaryDirectory() as tmp:
    _conflict_path = write_record(tmp, "conflict.json", record_conflict)
    result_conflict = bod.build_dataset([_conflict_path])
check("elongation conflict: produces zero modelling rows (no usable target)",
      len(result_conflict.dataset) == 0)
conflict_excluded = [e for e in result_conflict.excluded_records if e.get("observation_id") == "O2"]
check("elongation conflict: excluded with reason 'elongation_type_conflict_unresolved' (distinct from replicate_target_unresolved)",
      len(conflict_excluded) == 1 and conflict_excluded[0]["reason"] == "elongation_type_conflict_unresolved",
      detail=str(conflict_excluded))

# true replicate (two total-elongation values from the SAME slot, as Stage 3
# would already have merged them) still resolves via the existing
# unresolved-replicate path, untouched by the conflict logic
obs_true_replicate = make_observation(
    "O3", "M1", "G1",
    {"elongation_total": prop("elongation", None, elongation_type="total",
                               replicate_info={"is_aggregate": False, "n_replicates_reported": 2, "raw_replicate_values": [14.4, 14.9]})},
)
tr_replicate = bod.map_targets(obs_true_replicate)
check("true replicate: elongation_pct NaN via the pre-existing replicate-unresolved path, elongation_type_conflict stays False",
      _is_null(tr_replicate.values["elongation_pct"]) and tr_replicate.elongation_type_conflict is False
      and "elongation" in tr_replicate.replicate_unresolved_properties)

# ─────────────────────────────────────────────────────────────────────────
# Property scope / loading mode: an observation whose ONLY YS evidence is
# positively-incompatible (compressive and/or microscale/phase-level) must
# leave YS_MPa unresolved (NaN) -- never mapped to the bulk-tensile target
# -- but, mirroring the elongation-conflict precedent, provenance_verified
# stays True because real, Stage-3-verified evidence exists (it is just
# not eligible for THIS target). The exclusion reason must be the new,
# specific incompatible_property_scope / incompatible_loading_mode code,
# never generically lumped into "no_usable_target" (which means no
# evidence was found at all -- a materially different situation) or
# mislabeled as a provenance/hallucination failure.
# ─────────────────────────────────────────────────────────────────────────

# TEST K — pure loading-mode incompatibility (compressive-only YS evidence,
# no scope cue) -> incompatible_loading_mode
obs_compressive_only = make_observation(
    "O4", "M1", "G1",
    {"YS_unspecified_compressive": prop("YS", 277.0, property_scope="unspecified", loading_mode="compressive",
                                          source_column="Compressive yield strength /[MPa]",
                                          reported_unit="MPa", unit_verification_status="verified_from_table_header")},
)
tr_compressive = bod.map_targets(obs_compressive_only)
check("TEST K: YS_MPa left unresolved (NaN) -- never mapped from compressive-only evidence",
      _is_null(tr_compressive.values["YS_MPa"]), detail=str(tr_compressive.values))
check("TEST K: YS provenance_verified stays True (real evidence exists, just scope-incompatible)",
      tr_compressive.provenance_verified["YS_provenance_verified"] is True)
check("TEST K: incompatible_scope_targets flags YS_MPa as incompatible_loading_mode",
      tr_compressive.incompatible_scope_targets.get("YS_MPa") == "incompatible_loading_mode",
      detail=str(tr_compressive.incompatible_scope_targets))

# TEST L — pure property-scope incompatibility (phase-level, e.g. a
# micropillar ferrite/martensite strength value) -> incompatible_property_scope
obs_phase_only = make_observation(
    "O5", "M1", "G1",
    {"YS_phase_level_compressive": prop("YS", 149.0, property_scope="phase_level", loading_mode="compressive",
                                          source_column="Ferrite pillar compressive yield strength [MPa]",
                                          reported_unit="MPa", unit_verification_status="verified_from_table_header")},
)
tr_phase = bod.map_targets(obs_phase_only)
check("TEST L: YS_MPa left unresolved (NaN) -- never mapped from phase-level pillar evidence",
      _is_null(tr_phase.values["YS_MPa"]), detail=str(tr_phase.values))
check("TEST L: YS provenance_verified stays True",
      tr_phase.provenance_verified["YS_provenance_verified"] is True)
check("TEST L: incompatible_scope_targets flags YS_MPa as incompatible_property_scope (scope takes priority over mode)",
      tr_phase.incompatible_scope_targets.get("YS_MPa") == "incompatible_property_scope",
      detail=str(tr_phase.incompatible_scope_targets))

# TEST M — mixed observation: a normal bulk UTS value alongside
# incompatible-only YS evidence -> the row is still usable (UTS_MPa
# populated), YS_MPa simply stays NaN with its own distinct flag; this is
# NOT a whole-observation exclusion.
obs_mixed_m = make_observation(
    "O6", "M1", "G1",
    {"UTS": prop("UTS", 830.0, source_column="UTS [MPa]", reported_unit="MPa", unit_verification_status="verified_from_table_header"),
     "YS_unspecified_compressive": prop("YS", 277.0, property_scope="unspecified", loading_mode="compressive",
                                          source_column="Compressive yield strength /[MPa]",
                                          reported_unit="MPa", unit_verification_status="verified_from_table_header")},
)
tr_mixed_m = bod.map_targets(obs_mixed_m)
check("TEST M: UTS_MPa resolves normally (830.0), unaffected by YS's incompatibility",
      tr_mixed_m.values["UTS_MPa"] == 830.0)
check("TEST M: YS_MPa stays NaN, flagged incompatible_loading_mode",
      _is_null(tr_mixed_m.values["YS_MPa"]) and tr_mixed_m.incompatible_scope_targets.get("YS_MPa") == "incompatible_loading_mode")

# exclusion-reason integration check: an observation whose ONLY property is
# incompatible-scope YS gets the new specific reason, end-to-end through
# build_dataset (mirrors the elongation-conflict integration check above)
material_incompatible = make_material("M1", "TestGrade", [], [make_condition("C1", "baseline", [obs_compressive_only])])
record_incompatible = make_stage3_record("10.9999/scope-incompatible", [material_incompatible])
with tempfile.TemporaryDirectory() as tmp:
    _incompat_path = write_record(tmp, "incompatible.json", record_incompatible)
    result_incompatible = bod.build_dataset([_incompat_path])
check("TEST K integration: produces zero modelling rows (no usable target)",
      len(result_incompatible.dataset) == 0)
incompat_excluded = [e for e in result_incompatible.excluded_records if e.get("observation_id") == "O4"]
check("TEST K integration: excluded with reason 'incompatible_loading_mode' (distinct from no_usable_target/replicate_target_unresolved)",
      len(incompat_excluded) == 1 and incompat_excluded[0]["reason"] == "incompatible_loading_mode",
      detail=str(incompat_excluded))


print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
