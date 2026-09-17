"""
test_stage3_gate.py — offline, dependency-free sanity tests for Stage 3's
deterministic components and the observation-construction gate.

No LLM calls, no network access, no data/processed corpus files required for
the required 8 gate cases (they operate on synthetic source_context dicts
directly, per the approved test plan). A couple of extra checks at the
bottom exercise numeric_equivalent() and build_source_context() against a
tiny hand-built synthetic paper dict — still fully offline.

Run: python literature_pipeline/v2/extraction/test_stage3_gate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from literature_pipeline.v2.extraction import extract_observations as e3

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == 'FAIL' else ""))
    if not condition:
        FAILURES.append(label)


def ctx(material=None, processing=None, specimen=None, specimen_unique=False,
        baseline=False, tc_status="unresolved", tc_value=None, row_key=None,
        table_caption=None, section_context=None):
    return {
        "material_identifier": material,
        "processing_identifier": processing,
        "specimen_identifier": specimen,
        "specimen_identifier_unique_state": specimen_unique,
        "is_baseline_or_as_received": baseline,
        "test_condition": {"status": tc_status, "value": tc_value},
        "row_key": row_key,
        "table_caption": table_caption,
        "section_context": section_context,
    }


def item(prop, value, source_context):
    return {"property": prop, "value": value, "source_context": source_context}


# ─────────────────────────────────────────────────────────────────────────
# 1. Same-row UTS/YS merge
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 1147.5, ctx(material="DP780", row_key=("t0", 2), tc_status="confirmed_explicit", tc_value="RT")),
    item("YS", 820.0, ctx(material="DP780", row_key=("t0", 2), tc_status="confirmed_explicit", tc_value="RT")),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
check("1. same-row UTS/YS -> single valid_observation with both properties",
      len(results) == 1 and results[0]["observation_status"] == "valid_observation"
      and set(results[0]["properties"].keys()) == {"UTS", "YS"},
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 2. Same material, different processing condition -> no merge
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 1000.0, ctx(material="DP780", processing="IA700", row_key=("t1", 0))),
    item("YS", 700.0, ctx(material="DP780", processing="IA750", row_key=("t1", 1))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
check("2. same material, different processing -> two separate observations (not merged)",
      len(results) == 2
      and all(r["observation_status"] == "valid_observation" for r in results)
      and {list(r["properties"].keys())[0] for r in results} == {"UTS", "YS"},
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 3. Same material+processing, different test temperature -> no merge
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 900.0, ctx(material="DP780", processing="IA700", tc_status="confirmed_explicit", tc_value="RT", row_key=("t2", 0))),
    item("YS", 650.0, ctx(material="DP780", processing="IA700", tc_status="confirmed_explicit", tc_value="300C", row_key=("t2", 1))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
check("3. same material+processing, conflicting test temperature -> no merge",
      len(results) == 2 and all(r["observation_status"] == "valid_observation" for r in results),
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 4. Different tables, no positive condition identifier -> no merge
#    (and, correctly, neither becomes a valid observation on its own either)
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 1265.0, ctx(material="DP4", row_key=("t3", 0))),
    item("YS", 1470.0, ctx(material="DP4", row_key=("t7", 0))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
check("4. different tables, no positive condition identifier -> not merged, both orphan",
      len(results) == 2 and all(r["observation_status"] == "orphan_property_evidence" for r in results),
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 5. Explicit shared specimen identifier + compatible test condition -> merge
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 1300.0, ctx(material="DP4", specimen="RPS-2", specimen_unique=True,
                             tc_status="confirmed_explicit", tc_value="RT", row_key=("t4", 0))),
    item("YS", 1150.0, ctx(material="DP4", specimen="RPS-2", specimen_unique=True,
                            tc_status="confirmed_explicit", tc_value="RT", row_key=("t9", 3))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
check("5. shared unique specimen identifier + compatible test condition -> merged valid_observation",
      len(results) == 1 and results[0]["observation_status"] == "valid_observation"
      and set(results[0]["properties"].keys()) == {"UTS", "YS"},
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 6. Verified UTS + unsafe(unlinked) YS -> UTS-only valid observation,
#    YS preserved separately (not merged, not silently dropped)
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 980.0, ctx(material="DP780", processing="IA700", tc_status="confirmed_explicit", tc_value="RT", row_key=("t5", 0))),
    item("YS", 610.0, ctx(material="DP780", tc_status="unresolved", row_key=("t5", 1))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
valid = [r for r in results if r["observation_status"] == "valid_observation"]
orphan = [r for r in results if r["observation_status"] == "orphan_property_evidence"]
check("6. UTS safely linked -> valid_observation (UTS only); YS unresolved processing -> orphan, not merged",
      len(results) == 2 and len(valid) == 1 and list(valid[0]["properties"].keys()) == ["UTS"]
      and len(orphan) == 1 and orphan[0]["properties"][0]["property"] == "YS",
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 7. Orphan property with unresolved processing (material known)
# ─────────────────────────────────────────────────────────────────────────
items = [item("UTS", 1050.0, ctx(material="DP4", row_key=("t6", 0)))]
results = e3.construct_candidates(items, processing_state_type="not_established")
check("7. material known, processing unresolved -> orphan_property_evidence (not an observation)",
      len(results) == 1 and results[0]["observation_status"] == "orphan_property_evidence"
      and results[0]["linkage"]["material_identity"] == "confirmed"
      and results[0]["linkage"]["processing_identity"] == "unresolved",
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 8. Three replicate tensile rows -> one condition, replicate structure preserved
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 1140.0, ctx(material="DP4", processing="Q&T", tc_status="confirmed_explicit", tc_value="RT", row_key=("t8", 0))),
    item("UTS", 1152.0, ctx(material="DP4", processing="Q&T", tc_status="confirmed_explicit", tc_value="RT", row_key=("t8", 1))),
    item("UTS", 1148.0, ctx(material="DP4", processing="Q&T", tc_status="confirmed_explicit", tc_value="RT", row_key=("t8", 2))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
rep_info = results[0]["properties"]["UTS"]["replicate_info"] if results else None
check("8. three replicate rows -> ONE valid_observation, replicate values preserved unaggregated (not averaged, not 3 fake conditions)",
      len(results) == 1 and results[0]["observation_status"] == "valid_observation"
      and rep_info is not None and rep_info["n_replicates_reported"] == 3
      and rep_info["raw_replicate_values"] == [1140.0, 1152.0, 1148.0]
      and results[0]["properties"]["UTS"]["value"] is None,
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 9. BONUS — material identity alone (assumed_single_material) must never
#    be sufficient to pass the gate, even with confirmed processing.
# ─────────────────────────────────────────────────────────────────────────
items = [item("UTS", 900.0, ctx(material=None, processing="IA700", tc_status="confirmed_explicit", tc_value="RT", row_key=("t10", 0)))]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition",
                                   material_grade_label="DP780", n_materials_in_paper=1)
check("9. BONUS: assumed_single_material (no explicit identifier) does NOT pass the gate alone -> orphan",
      len(results) == 1 and results[0]["observation_status"] == "orphan_property_evidence"
      and results[0]["linkage"]["material_identity"] == "assumed_single_material",
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 10. BONUS heliyon-pattern regression check — standard tensile UTS vs.
#     RPS pre-strain YS, different tables, different processing labels ->
#     must split, must NOT reproduce the confirmed incorrect merge.
# ─────────────────────────────────────────────────────────────────────────
items = [
    item("UTS", 1265.0, ctx(material="DP4", processing="standard_tensile", tc_status="confirmed_explicit", tc_value="RT", row_key=("t1", 9))),
    item("YS", 1470.0, ctx(material="DP4", processing="RPS_pre_strain", tc_status="confirmed_explicit", tc_value="RT", row_key=("t3", 0))),
]
results = e3.construct_candidates(items, processing_state_type="explicit_processing_condition")
check("10. BONUS heliyon regression: different processing (standard vs RPS pre-strain) -> split, not merged",
      len(results) == 2 and all(r["observation_status"] == "valid_observation" for r in results)
      and {list(r["properties"].keys())[0] for r in results} == {"UTS", "YS"},
      detail=str(results))

# ─────────────────────────────────────────────────────────────────────────
# 11-14. PATCH REGRESSION TESTS (post-baseline-run): the ACTUAL heliyon
# pattern -- both items carry the group-level is_baseline_or_as_received
# flag (no explicit processing_identifier at all), which is exactly what
# the confirmed baseline-run bug exploited. Test 10 above used explicit
# conflicting processing_identifier labels, a DIFFERENT code path that
# would not have caught this. These four close that gap.
# ─────────────────────────────────────────────────────────────────────────

# 11. singleton + inherited group baseline flag + explicit contradictory
#     local processing/specimen state (different table caption elsewhere
#     in the group) -> must NOT resolve confirmed_baseline. This is the
#     real heliyon reproduction: no processing_identifier anywhere, only
#     the baseline flag and two different table captions.
heliyon_items = [
    item("UTS", 1265.0, ctx(material="DP4", baseline=True, tc_status="confirmed_explicit", tc_value="RT",
                             row_key=(1, 9), table_caption="Mechanical properties.")),
    item("YS", 1470.0, ctx(material="DP4", baseline=True, tc_status="confirmed_explicit", tc_value="RT",
                            row_key=(3, 0), table_caption="Mechanical properties of DP4 after pre-strain and backing tested by the RPS method.")),
]
heliyon_results = e3.construct_candidates(heliyon_items, processing_state_type="baseline_or_as_received_state",
                                           material_grade_label="DP4", n_materials_in_paper=1)
check("11. real heliyon pattern: baseline flag + contradictory captions -> neither singleton resolves confirmed_baseline",
      len(heliyon_results) == 2
      and all(r["linkage"]["processing_identity"] != "confirmed_baseline" for r in heliyon_results),
      detail=str(heliyon_results))
check("11b. both properties correctly become orphan_property_evidence (material known, processing unresolved) -- neither fabricated into a valid observation",
      all(r["observation_status"] == "orphan_property_evidence" for r in heliyon_results),
      detail=str(heliyon_results))

# 12. singleton + inherited baseline flag but NO location-specific baseline
#     evidence at all (no table_caption/section_context recorded anywhere
#     in the group) -> must not be promoted merely from the group prior.
no_evidence_items = [
    item("UTS", 900.0, ctx(material="DP4", baseline=True, tc_status="default_implicit", row_key=(0, 0))),
]
no_evidence_results = e3.construct_candidates(no_evidence_items, processing_state_type="baseline_or_as_received_state",
                                               material_grade_label="DP4", n_materials_in_paper=1)
check("12. singleton with baseline flag but zero context signature anywhere in the group -> NOT promoted (unresolved, not confirmed_baseline)",
      no_evidence_results[0]["linkage"]["processing_identity"] == "unresolved"
      and no_evidence_results[0]["observation_status"] == "orphan_property_evidence",
      detail=str(no_evidence_results))

# 13. singleton with positive local baseline/as-received evidence (the
#     group's cited evidence resolves to exactly ONE context signature) ->
#     MAY resolve confirmed_baseline. Purely structural (one signature,
#     not zero, not more) -- no keyword reading of the caption's content.
one_context_items = [
    item("YS", 700.0, ctx(material="DP780", baseline=True, tc_status="default_implicit", row_key=(0, 0),
                           table_caption="Mechanical properties of DP780 steel.")),
]
one_context_results = e3.construct_candidates(one_context_items, processing_state_type="baseline_or_as_received_state",
                                               material_grade_label="DP780", n_materials_in_paper=1)
check("13. singleton with exactly one context signature across the group -> MAY resolve confirmed_baseline -> valid_observation",
      one_context_results[0]["linkage"]["processing_identity"] == "confirmed_baseline"
      and one_context_results[0]["observation_status"] == "valid_observation",
      detail=str(one_context_results))

# 14. two different-location baseline-prior items -> still must not merge
#     without positive location-specific compatibility (direct test of the
#     processing_compatible() fix itself, independent of the singleton fix).
two_baseline_items = [
    item("UTS", 500.0, ctx(material="DP4", baseline=True, tc_status="default_implicit", row_key=(0, 0), table_caption="Table A.")),
    item("YS", 300.0, ctx(material="DP4", baseline=True, tc_status="default_implicit", row_key=(1, 0), table_caption="Table B.")),
]
two_baseline_results = e3.construct_candidates(two_baseline_items, processing_state_type="baseline_or_as_received_state",
                                                material_grade_label="DP4", n_materials_in_paper=1)
check("14. two different-location baseline-prior items -> never merged into one observation",
      len(two_baseline_results) == 2
      and all(len(r["properties"]) == 1 for r in two_baseline_results if r["observation_status"] == "valid_observation"),
      detail=str(two_baseline_results))

# ─────────────────────────────────────────────────────────────────────────
# Deterministic-component checks (numeric equivalence + source_context)
# ─────────────────────────────────────────────────────────────────────────
check("numeric_equivalent: '765' == '765.0'", e3.numeric_equivalent("765", "765.0"))
check("numeric_equivalent: '765' != '766'", not e3.numeric_equivalent("765", "766"))
check("numeric_equivalent: '1,200' == '1200'", e3.numeric_equivalent("1,200", "1200"))
check("numeric_equivalent: '830 MPa' handled via parse (fails, non-numeric remainder) -> False",
      not e3.numeric_equivalent("830 MPa", "830"))

SYN_PAPER = {
    "doi": "10.9999/synthetic",
    "tables": [
        {"caption": "Table 1: Mechanical properties of DP780 steel", "headers": ["Specimen", "TS (MPa)", "YS (MPa)"],
         "rows": [["S1", "1147.5", "820"]]},
    ],
    "sections": [],
}
sc = e3.build_source_context(SYN_PAPER, {"type": "table", "table_index": 0, "row_index": 0, "column": "TS (MPa)"},
                              material_grade_label="DP780", processing_state_type="explicit_processing_condition",
                              group_has_test_condition_evidence=False)
check("build_source_context: material_identifier recovered from caption mention ('DP780')",
      sc["material_identifier"] == "DP780", detail=str(sc))
check("build_source_context: specimen_identifier recovered from 'Specimen' column ('S1')",
      sc["specimen_identifier"] == "S1", detail=str(sc))
check("build_source_context: no test-condition columns + no group test_condition_evidence -> default_implicit",
      sc["test_condition"]["status"] == "default_implicit", detail=str(sc))

resolved = e3.read_location_value(SYN_PAPER, {"type": "table", "table_index": 0, "row_index": 0, "column": "TS (MPa)"})
check("read_location_value: deterministic table-cell read returns '1147.5' (zero LLM calls)",
      resolved == "1147.5", detail=str(resolved))

reverify = e3.reverify_property_item(SYN_PAPER, {"type": "table", "table_index": 0, "row_index": 0, "column": "TS (MPa)"}, "1147.5")
check("reverify_property_item: exact match verifies", reverify["stage3_verified"] is True, detail=str(reverify))
reverify2 = e3.reverify_property_item(SYN_PAPER, {"type": "table", "table_index": 0, "row_index": 0, "column": "TS (MPa)"}, "1147.50")
check("reverify_property_item: numeric-equivalent '1147.50' vs '1147.5' still verifies (recovers Stage2-style formatting false negative)",
      reverify2["stage3_verified"] is True, detail=str(reverify2))
reverify3 = e3.reverify_property_item(SYN_PAPER, {"type": "table", "table_index": 0, "row_index": 0, "column": "TS (MPa)"}, "9999")
check("reverify_property_item: fabricated/wrong value correctly fails verification",
      reverify3["stage3_verified"] is False, detail=str(reverify3))

comp = e3.extract_composition_from_table(
    {"tables": [{"caption": "Table 0: Composition", "headers": ["C", "Mn", "Si", "Fe"], "rows": [["0.17", "2.1", "0.3", "Balance"]]}]},
    {"type": "table", "table_index": 0, "row_index": None, "column": None},
)
check("extract_composition_from_table: reads C/Mn/Si values and recognizes 'Balance' qualifier for Fe",
      comp is not None and any(e["element"] == "C" and e["value"] == 0.17 for e in comp)
      and any(e["element"] == "Fe" and e["qualifier"] == "balance" for e in comp),
      detail=str(comp))

# ─────────────────────────────────────────────────────────────────────────
# LOCAL-CONTEXT LAYER TESTS (build_local_bundle, verify_llm_source_span,
# normalize/verify_and_filter_llm_output, merge_llm_context_into_source_
# context, process_llm_response_for_group, assemble_processing_conditions).
# No network access anywhere below.
# ─────────────────────────────────────────────────────────────────────────

LOCAL_PAPER = {
    "doi": "10.9999/local-context-synthetic",
    "tables": [],
    "sections": [
        {"heading": "Materials", "paragraphs": [
            "Steels of two grades were investigated in this work.",           # 0
            "The DP780 steel used in this study was received as hot-rolled coil.",  # 1
            "A separate reference alloy, HSLA450, was also examined for comparison.",  # 2
        ]},
        {"heading": "Results", "paragraphs": [
            "Tensile tests were carried out at room temperature.",           # 0
            "After quenching and partitioning, the Q&T steel exhibits a yield strength of 1465 MPa and an ultimate tensile strength of 1952 MPa.",  # 1
            "These values are consistent with prior reports on similar alloys.",  # 2
        ]},
    ],
}

# a. fabricated span exists elsewhere in the bundle, not at the claimed location -> fail
check("a. fabricated span (real text, wrong location) fails location-specific verification",
      e3.verify_llm_source_span(
          LOCAL_PAPER,
          {"type": "section", "heading": "Materials", "paragraph_index": 2},  # claims paragraph 2 ("HSLA450"...)
          "the DP780 steel used in this study",                              # but this text is actually paragraph 1
      ) is False)

# b. genuine span at the claimed location -> pass
check("b. genuine span at its true claimed location verifies",
      e3.verify_llm_source_span(
          LOCAL_PAPER,
          {"type": "section", "heading": "Materials", "paragraph_index": 1},
          "the DP780 steel used in this study",
      ) is True)

# c. deterministic context cannot be overwritten by LLM output
det_ctx = ctx(material="DP780", row_key=("t0", 0))  # already deterministically resolved
merged = e3.merge_llm_context_into_source_context(
    det_ctx, {"material_identifier": "SomeOtherGrade", "processing_identifier": None,
              "confirmed_unique_specimen_identifiers": set()})
check("c. an already-populated deterministic material_identifier is NOT overwritten by LLM output",
      merged["material_identifier"] == "DP780", detail=str(merged))

# d. verified LLM material identifier upgrades an otherwise assumed_single_material context
null_ctx = ctx(material=None, processing="IA700", tc_status="confirmed_explicit", tc_value="RT", row_key=("t9", 0))
before = e3.construct_candidates([item("UTS", 900.0, null_ctx)], processing_state_type="explicit_processing_condition",
                                  material_grade_label="DP780", n_materials_in_paper=1)
check("d0. baseline: no identifier -> assumed_single_material -> orphan (sanity check before upgrade)",
      before[0]["linkage"]["material_identity"] == "assumed_single_material"
      and before[0]["observation_status"] == "orphan_property_evidence")

upgraded_ctx = e3.merge_llm_context_into_source_context(
    null_ctx, {"material_identifier": "DP780", "processing_identifier": None, "confirmed_unique_specimen_identifiers": set()})
after = e3.construct_candidates([item("UTS", 900.0, upgraded_ctx)], processing_state_type="explicit_processing_condition")
check("d. verified LLM material_identifier upgrades assumed_single_material -> confirmed -> valid_observation",
      after[0]["linkage"]["material_identity"] == "confirmed" and after[0]["observation_status"] == "valid_observation",
      detail=str(after))

# e. an unverified identifier cannot upgrade it (verification fails upstream -> never reaches the merge at all)
mr_unverified = e3._normalize_resolution({
    "identifier": "DP780", "status": "confirmed_in_local_text",
    "source_span": "text that does not actually appear anywhere in the bundle",
    "location": {"type": "section", "heading": "Materials", "paragraph_index": 1},
})
bundle_for_e = {"allowed_locations": {("section", "Materials", 1)}}
verified_ok = e3._location_allowed(bundle_for_e, mr_unverified["location"]) and e3.verify_llm_source_span(
    LOCAL_PAPER, mr_unverified["location"], mr_unverified["source_span"])
check("e. an LLM-claimed identifier whose span doesn't verify is never promoted to verified_material_identifier",
      verified_ok is False)

# f. two groups sharing the same processing identifier share one ProcessingCondition,
#    but remain separate Observations because test conditions differ
obs1 = e3.build_valid_observation(
    [item("UTS", 1000.0, ctx(material="DP780", processing="IA700", tc_status="confirmed_explicit", tc_value="RT", row_key=("ta", 0)))],
    "confirmed", "confirmed_explicit")
obs2 = e3.build_valid_observation(
    [item("YS", 700.0, ctx(material="DP780", processing="IA700", tc_status="confirmed_explicit", tc_value="300C", row_key=("tb", 0)))],
    "confirmed", "confirmed_explicit")
conditions = e3.assemble_processing_conditions("M1", [("G1", obs1), ("G2", obs2)])
check("f. same processing identifier (IA700) from two different groups -> ONE ProcessingCondition, TWO Observations",
      len(conditions) == 1 and len(conditions[0]["observations"]) == 2
      and conditions[0]["observations"][0]["observation_id"] != conditions[0]["observations"][1]["observation_id"]
      and {o["stage2_evidence_group_ref"] for o in conditions[0]["observations"]} == {"G1", "G2"},
      detail=str(conditions))

# g. prose numeric token parsing and provenance — the literal example from the design round
PROSE_LOC = {"type": "section", "heading": "Results", "paragraph_index": 1}
prose_bundle = {"allowed_locations": {("section", "Results", 1)}}
raw_llm_json = {
    "material_identifier_resolution": None,
    "processing_identifier_resolution": None,
    "stage3_processing_state_type": "explicit_processing_condition",
    "processing_route": [],
    "test_condition_identifiers": [],
    "prose_property_candidates": [
        {"property": "YS", "reported_property_name": "yield strength", "reported_numeric_token": "1465",
         "reported_unit": "MPa", "source_span": "yield strength of 1465 MPa", "location": PROSE_LOC},
        {"property": "UTS", "reported_property_name": "ultimate tensile strength", "reported_numeric_token": "1952",
         "reported_unit": "MPa", "source_span": "ultimate tensile strength of 1952 MPa", "location": PROSE_LOC},
    ],
    "linkage_cues": [], "additional_local_condition_candidates": [],
    "notes": "", "confidence": "high",
}
normalized_g = e3.normalize_llm_output(raw_llm_json)
verified_g = e3.verify_and_filter_llm_output(LOCAL_PAPER, prose_bundle, normalized_g)
check("g. prose YS candidate: span verifies at its exact location and numeric token parses to 1465.0",
      len(verified_g["verified_prose_properties"]) == 2
      and {c["property"]: c["value"] for c in verified_g["verified_prose_properties"]} == {"YS": 1465.0, "UTS": 1952.0},
      detail=str(verified_g["verified_prose_properties"]))
check("g2. prose candidate preserves reported_numeric_token/unit/source_span alongside the parsed value",
      verified_g["verified_prose_properties"][0]["reported_numeric_token"] == "1465"
      and verified_g["verified_prose_properties"][0]["reported_unit"] == "MPa"
      and verified_g["verified_prose_properties"][0]["source_span"] == "yield strength of 1465 MPa")

# g3. a fabricated prose numeric token (non-numeric) must be rejected, not silently coerced
raw_llm_bad = dict(raw_llm_json)
raw_llm_bad["prose_property_candidates"] = [
    {"property": "YS", "reported_property_name": "yield strength", "reported_numeric_token": "approximately 1465ish",
     "reported_unit": "MPa", "source_span": "yield strength of 1465 MPa", "location": PROSE_LOC},
]
verified_bad = e3.verify_and_filter_llm_output(LOCAL_PAPER, prose_bundle, e3.normalize_llm_output(raw_llm_bad))
check("g3. unparseable numeric token -> rejected_prose_properties, not silently coerced into a value",
      len(verified_bad["verified_prose_properties"]) == 0
      and len(verified_bad["rejected_prose_properties"]) == 1
      and verified_bad["rejected_prose_properties"][0]["reason"] == "unsupported_numerical_token")

# h. malformed JSON / timeout produces no valid observation
r_timeout = e3.process_llm_response_for_group(LOCAL_PAPER, prose_bundle, None, call_error="request failed: ReadTimeout")
r_malformed = e3.process_llm_response_for_group(LOCAL_PAPER, prose_bundle, "this is not json at all", None)
check("h. simulated timeout -> stage3_llm_status='timeout', no verified_output",
      r_timeout["stage3_llm_status"] == "timeout" and r_timeout["verified_output"] is None)
check("h2. malformed JSON -> stage3_llm_status='malformed_json', no verified_output",
      r_malformed["stage3_llm_status"] == "malformed_json" and r_malformed["verified_output"] is None)

# h3. process_evidence_group with a forced LLM failure (monkeypatched, no network) still
#     produces no phantom valid_observation and correctly logs a failure record
_ORIGINAL_RUN_LLM = e3.run_llm_extraction_pass


def _forced_failure(paper, material, group, bundle):
    return {"stage3_llm_status": "timeout", "verified_output": None, "raw_error": "simulated for offline test"}


FAILURE_PAPER = {
    "doi": "10.9999/failure-synthetic",
    "tables": [],
    "sections": [{"heading": "Results", "paragraphs": ["The steel exhibits a yield strength of 900 MPa."]}],
}
FAILURE_MATERIAL = {"material_id": "M1", "grade_label": "DP780", "composition_evidence": [], "evidence_groups": []}
FAILURE_GROUP = {
    "group_id": "G1", "processing_state_type": "explicit_processing_condition",
    "processing_evidence": [], "test_condition_evidence": [],
    "property_evidence": [{"property": "YS", "location": {"type": "section", "heading": "Results", "paragraph_index": 0},
                            "quote": "yield strength of 900 MPa"}],
}
e3.run_llm_extraction_pass = _forced_failure
try:
    result_h3 = e3.process_evidence_group(FAILURE_PAPER, FAILURE_MATERIAL, FAILURE_GROUP, n_materials_in_paper=1, llm_enabled=True)
finally:
    e3.run_llm_extraction_pass = _ORIGINAL_RUN_LLM
check("h3. forced LLM failure -> no valid_observation fabricated, failure recorded, prose property unresolved not orphaned-as-fake",
      len(result_h3["valid_observations"]) == 0 and len(result_h3["failures"]) == 1
      and result_h3["failures"][0]["type"] == "timeout",
      detail=str(result_h3))

# i. local discovery cannot escape the defined paragraph/table window
DISCOVERY_PAPER = {
    "doi": "10.9999/discovery-synthetic",
    "tables": [{"caption": "Table 0", "headers": ["A"], "rows": [["1"]]},
               {"caption": "Table 1", "headers": ["B"], "rows": [["2"]]}],
    "sections": [{"heading": "Results", "paragraphs": [f"paragraph {i} text about condition {i}." for i in range(10)]}],
}
# Bundle was built around table 0 and paragraph 5 (+/-2 -> paragraphs 3-7) only
discovery_bundle = {
    "allowed_locations": {("table", 0)} | {("section", "Results", p) for p in range(3, 8)},
}
raw_discovery = {
    "material_identifier_resolution": None, "processing_identifier_resolution": None,
    "stage3_processing_state_type": "explicit_processing_condition",
    "processing_route": [], "test_condition_identifiers": [], "prose_property_candidates": [],
    "linkage_cues": [],
    "additional_local_condition_candidates": [
        {"reason": "in-window, should be accepted", "process_type": "tempering",
         "source_span": "paragraph 5 text about condition 5.",
         "location": {"type": "section", "heading": "Results", "paragraph_index": 5}},
        {"reason": "out-of-window paragraph, must be rejected", "process_type": "tempering",
         "source_span": "paragraph 9 text about condition 9.",
         "location": {"type": "section", "heading": "Results", "paragraph_index": 9}},
        {"reason": "different table never cited by this group, must be rejected", "process_type": "tempering",
         "source_span": "2", "location": {"type": "table", "table_index": 1, "row_index": 0, "column": "B"}},
    ],
    "notes": "", "confidence": "medium",
}
verified_i = e3.verify_and_filter_llm_output(DISCOVERY_PAPER, discovery_bundle, e3.normalize_llm_output(raw_discovery))
check("i. in-window discovery candidate (paragraph 5) is accepted",
      len(verified_i["verified_discoveries"]) == 1
      and verified_i["verified_discoveries"][0]["location"]["paragraph_index"] == 5,
      detail=str(verified_i["verified_discoveries"]))
check("i2. out-of-window paragraph (9) and never-cited table (index 1) are both rejected, not silently accepted",
      len(verified_i["rejected_discoveries"]) == 2
      and {d["location"].get("paragraph_index") for d in verified_i["rejected_discoveries"] if d["location"]["type"] == "section"} == {9}
      and any(d["location"]["type"] == "table" for d in verified_i["rejected_discoveries"]),
      detail=str(verified_i["rejected_discoveries"]))

# bonus: build_local_bundle itself produces exactly this bounded window (end-to-end, no LLM)
BUNDLE_GROUP = {
    "group_id": "G1", "processing_state_type": "explicit_processing_condition",
    "processing_evidence": [], "test_condition_evidence": [],
    "property_evidence": [{"property": "YS", "location": {"type": "section", "heading": "Results", "paragraph_index": 5},
                            "quote": "..."}],
}
BUNDLE_MATERIAL = {"material_id": "M1", "grade_label": "X", "composition_evidence": [], "evidence_groups": [BUNDLE_GROUP]}
real_bundle = e3.build_local_bundle(DISCOVERY_PAPER, BUNDLE_MATERIAL, BUNDLE_GROUP)
check("bonus: build_local_bundle window for a paragraph-5 citation is exactly paragraphs 3-7 (never reaches 9, never reaches table 1)",
      real_bundle["allowed_locations"] == {("section", "Results", p) for p in range(3, 8)},
      detail=str(real_bundle["allowed_locations"]))

# ─────────────────────────────────────────────────────────────────────────
# End-to-end assembly smoke test (deterministic only, llm_enabled=False):
# Paper -> Material -> ProcessingCondition -> Observation, two evidence
# groups under one material with different processing conditions.
# ─────────────────────────────────────────────────────────────────────────
ASSEMBLY_PAPER = {
    "doi": "10.9999/integration-smoke",
    "tables": [
        {"caption": "Table 1: Mechanical properties of DP780 steel", "headers": ["Specimen", "Condition", "TS (MPa)", "YS (MPa)"],
         "rows": [["S1", "IA700", "1147.5", "820"], ["S2", "IA750", "1050", "780"]]},
        {"caption": "Table 0: Composition of DP780", "headers": ["C", "Mn", "Si", "Fe"], "rows": [["0.17", "2.1", "0.3", "Balance"]]},
    ],
    "sections": [],
}
ASSEMBLY_STAGE2 = {
    "candidate_materials": [{
        "material_id": "M1", "grade_label": "DP780",
        "composition_evidence": [{"location": {"type": "table", "table_index": 1, "row_index": None, "column": None}, "quote": "0.17"}],
        "evidence_groups": [
            {"group_id": "G1", "processing_state_type": "explicit_processing_condition",
             "processing_evidence": [], "test_condition_evidence": [],
             "property_evidence": [
                 {"property": "UTS", "location": {"type": "table", "table_index": 0, "row_index": 0, "column": "TS (MPa)"}, "quote": "1147.5"},
                 {"property": "YS", "location": {"type": "table", "table_index": 0, "row_index": 0, "column": "YS (MPa)"}, "quote": "820"},
             ]},
            {"group_id": "G2", "processing_state_type": "explicit_processing_condition",
             "processing_evidence": [], "test_condition_evidence": [],
             "property_evidence": [
                 {"property": "UTS", "location": {"type": "table", "table_index": 0, "row_index": 1, "column": "TS (MPa)"}, "quote": "1050"},
             ]},
        ],
    }],
}
paper_record = e3.assemble_paper_record("10.9999/integration-smoke", ASSEMBLY_STAGE2, ASSEMBLY_PAPER, llm_enabled=False)
m1 = paper_record["materials"][0]
check("assembly: composition attached once to the material, 4 elements incl. Fe=balance",
      m1["composition"]["status"] == "reported" and len(m1["composition"]["elements"]) == 4)
check("assembly: two different processing identifiers (IA700/IA750) -> two separate ProcessingCondition nodes",
      len(m1["processing_conditions"]) == 2
      and {c["processing_key"] for c in m1["processing_conditions"]} == {"ia700", "ia750"})
check("assembly: G1's row (UTS+YS same row) -> one two-property observation under M1-P1",
      m1["processing_conditions"][0]["observations"][0]["property_completeness"]["completeness_tier"] == "two_properties")
check("assembly: G2's row -> one single-property observation under its own condition, stable IDs assigned",
      m1["processing_conditions"][1]["observations"][0]["observation_id"] == "M1-P2-O1"
      and m1["processing_conditions"][1]["observations"][0]["stage2_evidence_group_ref"] == "G2")

# ─────────────────────────────────────────────────────────────────────────
# PATCH TESTS (post-baseline-run): crash fix, empty-bundle skip, hard timeout
# ─────────────────────────────────────────────────────────────────────────

# j. out-of-range table_index (dt-2022-10-010's exact pattern: only 3 tables,
#    Stage 2 cited table_index=6) must not crash build_local_bundle, must be
#    recorded in invalid_locations, must not appear in allowed_locations.
CRASH_PAPER = {
    "doi": "10.9999/crash-synthetic",
    "tables": [{"caption": "Table 0", "headers": ["A"], "rows": [["1"]]},
               {"caption": "Table 1", "headers": ["B"], "rows": [["2"]]},
               {"caption": "Table 2", "headers": ["C"], "rows": [["3"]]}],
    "sections": [],
}
CRASH_MATERIAL = {"material_id": "M1", "grade_label": "X", "composition_evidence": [], "evidence_groups": []}
CRASH_GROUP = {
    "group_id": "G1", "processing_state_type": "baseline_or_as_received_state",
    "processing_evidence": [], "test_condition_evidence": [],
    "property_evidence": [{"property": "UTS", "location": {"type": "table", "table_index": 6, "row_index": 4, "column": "TS (MPa)"},
                            "quote": "999"}],
}
try:
    crash_bundle = e3.build_local_bundle(CRASH_PAPER, CRASH_MATERIAL, CRASH_GROUP)
    crashed = False
except IndexError:
    crashed = True
check("j. out-of-range table_index does NOT crash build_local_bundle (dt-2022-10-010 fix)",
      not crashed)
check("j2. the invalid location is recorded in invalid_locations, not silently dropped",
      not crashed and len(crash_bundle["invalid_locations"]) == 1
      and crash_bundle["invalid_locations"][0]["table_index"] == 6)
check("j3. the invalid table never appears in allowed_locations",
      not crashed and ("table", 6) not in crash_bundle["allowed_locations"])

# k. a group whose only location is invalid -> bundle_has_usable_content is
#    False -> the LLM must never be called (verified via a call counter),
#    and an insufficient_local_evidence failure is recorded.
_call_count = {"n": 0}
_ORIGINAL_RUN_LLM_2 = e3.run_llm_extraction_pass


def _counting_run_llm(paper, material, group, bundle):
    _call_count["n"] += 1
    return _ORIGINAL_RUN_LLM_2(paper, material, group, bundle)


e3.run_llm_extraction_pass = _counting_run_llm
try:
    result_k = e3.process_evidence_group(CRASH_PAPER, CRASH_MATERIAL, CRASH_GROUP, n_materials_in_paper=1, llm_enabled=True)
finally:
    e3.run_llm_extraction_pass = _ORIGINAL_RUN_LLM_2
check("k. all-invalid-location group -> LLM never called",
      _call_count["n"] == 0)
check("k2. insufficient_local_evidence failure recorded instead",
      len(result_k["failures"]) == 1 and result_k["failures"][0]["type"] == "insufficient_local_evidence")
check("k3. invalid location also surfaced as a rejected candidate with an explicit reason",
      any(r.get("reason") == "invalid_stage2_location" for r in result_k["rejected_candidates"]))

# l. jmrt-2020-06-063's exact pattern: a 0-paragraph section citation ->
#    validate_location already returns False for it (0 <= 0 < 0 is False)
#    -> invalid, excluded, bundle ends up unusable, LLM skipped.
EMPTY_SECTION_PAPER = {
    "doi": "10.9999/empty-section-synthetic",
    "tables": [],
    "sections": [{"heading": "Mechanical performance", "paragraphs": []}],
}
EMPTY_SECTION_GROUP = {
    "group_id": "G1", "processing_state_type": "explicit_processing_condition",
    "processing_evidence": [], "test_condition_evidence": [],
    "property_evidence": [{"property": "UTS", "location": {"type": "section", "heading": "Mechanical performance", "paragraph_index": 0},
                            "quote": "some quote"}],
}
empty_bundle = e3.build_local_bundle(EMPTY_SECTION_PAPER, CRASH_MATERIAL, EMPTY_SECTION_GROUP)
check("l. 0-paragraph section citation is caught by validate_location and treated as invalid",
      len(empty_bundle["invalid_locations"]) == 1)
check("l2. bundle_has_usable_content correctly returns False for this bundle",
      e3.bundle_has_usable_content(empty_bundle) is False)

# m. hard timeout mechanism: caller-side deadline is genuine, offline (a
#    fake slow worker substitutes for the real Ollama HTTP call — no
#    network involved). Asserts BOTH that call_llm raises at approximately
#    the configured timeout, AND that the calling code's own elapsed time
#    is close to the timeout, not the fake worker's full duration -- this
#    is the direct proof the previous ThreadPoolExecutor-based mechanism
#    lacked (it silently waited out the full worker duration instead).
import time as _time

_original_post = e3._post_to_ollama


def _slow_post(prompt):
    _time.sleep(3)
    class FakeResp:
        status_code = 200
        def json(self):
            return {"response": "{}"}
    return FakeResp()


_original_hard_timeout = e3.LLM_WALLCLOCK_HARD_TIMEOUT_S
e3._post_to_ollama = _slow_post
e3.LLM_WALLCLOCK_HARD_TIMEOUT_S = 1
_t0 = _time.time()
try:
    e3.call_llm("irrelevant prompt")
    m_raised = False
except e3.LLMCallError as _e:
    m_raised = True
    m_message = str(_e)
_m_elapsed = _time.time() - _t0
e3._post_to_ollama = _original_post
e3.LLM_WALLCLOCK_HARD_TIMEOUT_S = _original_hard_timeout

check("m. caller-side hard timeout: call_llm raises LLMCallError (not a successful return)",
      m_raised)
check("m2. caller-side hard timeout: the CALLER's own elapsed time is close to the 1s deadline, not the fake worker's 3s duration",
      m_raised and _m_elapsed < 2.0, detail=f"elapsed={_m_elapsed:.2f}s")
check("m3. timeout is recorded explicitly and described as caller-side / non-cancelling",
      m_raised and "caller-side" in m_message and "not cancelled server-side" in m_message)

# ─────────────────────────────────────────────────────────────────────────
# PATCH (v0.4): property-unit provenance — purely additive. Tests 1-11
# exercise the two new pure helpers directly; 12-20 confirm the new fields
# are wired into build_valid_observation/build_orphan_evidence WITHOUT
# altering value/status/linkage/clustering behaviour. The full suite above
# (58 pre-existing checks) was re-run completely unmodified and still
# passes, which is itself the strongest evidence this patch changed nothing
# about any existing decision.
# ─────────────────────────────────────────────────────────────────────────

check("unit1. table header 'UTS (MPa)' -> verified_from_table_header, MPa",
      e3.resolve_property_unit(source_column="UTS (MPa)") ==
      {"source_column": "UTS (MPa)", "reported_unit": "MPa", "unit_verification_status": "verified_from_table_header"})

check("unit2. table header 'Yield strength, MPa' -> MPa",
      e3.resolve_property_unit(source_column="Yield strength, MPa")["reported_unit"] == "MPa")

check("unit3. table header 'Elongation (%)' -> %",
      e3.resolve_property_unit(source_column="Elongation (%)")["reported_unit"] == "%")

check("unit4. table header with no unit ('UTS') -> reported_unit=None, not_available",
      e3.resolve_property_unit(source_column="UTS") ==
      {"source_column": "UTS", "reported_unit": None, "unit_verification_status": "not_available"})

check("unit5. table header with unsupported unit family ('UTS (ksi)') -> reported_unit=None, not_available "
      "(ksi deliberately not implemented in this patch)",
      e3.resolve_property_unit(source_column="UTS (ksi)")["reported_unit"] is None)

check("unit6. GPa recognized", e3._detect_unit_token("Elastic modulus (GPa)") == "GPa")

check("unit7. Pa recognized without falsely matching inside MPa/GPa",
      e3._detect_unit_token("Pressure (Pa)") == "Pa"
      and e3._detect_unit_token("Strength (MPa)") == "MPa"
      and e3._detect_unit_token("Modulus (GPa)") == "GPa")

check("unit8. 'percent' normalizes to '%'", e3._detect_unit_token("Elongation (percent)") == "%")

check("unit9. prose: model claims MPa AND source_span contains MPa -> verified_from_local_text",
      e3.resolve_property_unit(prose_reported_unit="MPa", prose_source_span="yield strength of 1465 MPa") ==
      {"source_column": None, "reported_unit": "MPa", "unit_verification_status": "verified_from_local_text"})

check("unit10. prose: model claims MPa but source_span does NOT contain MPa -> reported_by_model_unverified, "
      "reported_unit stays None (unverified claim never smuggled through)",
      e3.resolve_property_unit(prose_reported_unit="MPa", prose_source_span="yield strength of 1465") ==
      {"source_column": None, "reported_unit": None, "unit_verification_status": "reported_by_model_unverified"})

check("unit11. prose: no unit claim at all -> not_available",
      e3.resolve_property_unit(prose_reported_unit=None, prose_source_span="yield strength of 1465 MPa") ==
      {"source_column": None, "reported_unit": None, "unit_verification_status": "not_available"})

_UNIT_TEST_CTX = {
    "material_identifier": "DP780", "processing_identifier": "IA700", "specimen_identifier": None,
    "specimen_identifier_unique_state": False, "is_baseline_or_as_received": False,
    "test_condition": {"status": "default_implicit", "value": None},
    "row_key": (0, 0), "table_caption": "Mechanical properties", "section_context": None,
}
_item_with_unit = {"property": "UTS", "value": 980.0, "source_context": _UNIT_TEST_CTX,
                    "unit_info": {"source_column": "UTS (MPa)", "reported_unit": "MPa",
                                  "unit_verification_status": "verified_from_table_header"}}

_valid_obs_u = e3.build_valid_observation([_item_with_unit], "confirmed", "confirmed_explicit")
check("unit12. unit metadata appears on a valid_observation property record",
      _valid_obs_u["properties"]["UTS"]["reported_unit"] == "MPa"
      and _valid_obs_u["properties"]["UTS"]["source_column"] == "UTS (MPa)"
      and _valid_obs_u["properties"]["UTS"]["unit_verification_status"] == "verified_from_table_header")

_orphan_obs_u = e3.build_orphan_evidence([_item_with_unit], "assumed_single_material", "unresolved")
check("unit13. unit metadata appears on an orphan_property_evidence property record",
      _orphan_obs_u["properties"][0]["reported_unit"] == "MPa"
      and _orphan_obs_u["properties"][0]["source_column"] == "UTS (MPa)")

check("unit14. numeric property value is unchanged by the patch", _valid_obs_u["properties"]["UTS"]["value"] == 980.0)
check("unit15. observation_status is unchanged (still valid_observation)",
      _valid_obs_u["observation_status"] == "valid_observation")
check("unit16. material_identity is unchanged", _valid_obs_u["linkage"]["material_identity"] == "confirmed")
check("unit17. processing_identity is unchanged", _valid_obs_u["linkage"]["processing_identity"] == "confirmed_explicit")

_item_a = {"property": "UTS", "value": 980.0, "source_context": dict(_UNIT_TEST_CTX),
           "unit_info": {"source_column": "UTS (MPa)", "reported_unit": "MPa", "unit_verification_status": "verified_from_table_header"}}
_item_b = {"property": "YS", "value": 720.0, "source_context": dict(_UNIT_TEST_CTX),
           "unit_info": {"source_column": "Yield (ksi)", "reported_unit": None, "unit_verification_status": "not_available"}}
_clusters_with_units = e3.cluster_property_evidence([_item_a, _item_b])
_item_a_no_unit = {k: v for k, v in _item_a.items() if k != "unit_info"}
_item_b_no_unit = {k: v for k, v in _item_b.items() if k != "unit_info"}
_clusters_without_units = e3.cluster_property_evidence([_item_a_no_unit, _item_b_no_unit])
check("unit18. clustering result is identical whether unit_info is present, absent, or disagrees "
      "(unit metadata is never an input to positively_compatible)",
      len(_clusters_with_units) == 1 and len(_clusters_with_units[0]) == 2
      and len(_clusters_without_units) == 1 and len(_clusters_without_units[0]) == 2)

_rep_item_1 = {"property": "UTS", "value": 1200.0, "source_context": _UNIT_TEST_CTX,
               "unit_info": {"source_column": "UTS (MPa)", "reported_unit": "MPa", "unit_verification_status": "verified_from_table_header"}}
_rep_item_2 = {"property": "UTS", "value": 1210.0, "source_context": _UNIT_TEST_CTX,
               "unit_info": {"source_column": "UTS (MPa)", "reported_unit": "MPa", "unit_verification_status": "verified_from_table_header"}}
_valid_rep = e3.build_valid_observation([_rep_item_1, _rep_item_2], "confirmed", "confirmed_explicit")
check("unit19. replicate items agreeing on unit -> unit metadata preserved (never silently dropped)",
      _valid_rep["properties"]["UTS"]["reported_unit"] == "MPa"
      and _valid_rep["properties"]["UTS"]["unit_verification_status"] == "verified_from_table_header"
      and _valid_rep["properties"]["UTS"]["replicate_info"]["raw_replicate_values"] == [1200.0, 1210.0])

_rep_item_3 = {"property": "UTS", "value": 1210.0, "source_context": _UNIT_TEST_CTX,
               "unit_info": {"source_column": "TS (GPa)", "reported_unit": "GPa", "unit_verification_status": "verified_from_table_header"}}
_valid_rep_conflict = e3.build_valid_observation([_rep_item_1, _rep_item_3], "confirmed", "confirmed_explicit")
check("unit20. replicate items disagreeing on unit -> no unit silently chosen "
      "(reported_unit=None, unit_verification_status=not_available), raw values still preserved",
      _valid_rep_conflict["properties"]["UTS"]["reported_unit"] is None
      and _valid_rep_conflict["properties"]["UTS"]["unit_verification_status"] == "not_available"
      and _valid_rep_conflict["properties"]["UTS"]["replicate_info"]["raw_replicate_values"] == [1200.0, 1210.0])

# ─────────────────────────────────────────────────────────────────────────
# Elongation semantics: TEST G (uniform+total must NOT merge as replicates)
# and TEST H (true same-type replicates still merge exactly as before).
# ─────────────────────────────────────────────────────────────────────────

_elong_uniform_item = {"property": "elongation", "value": 8.8, "source_context": _UNIT_TEST_CTX,
                       "unit_info": {"source_column": "UE [%]", "reported_unit": "%", "unit_verification_status": "verified_from_table_header"},
                       "elongation_type": "uniform"}
_elong_total_item = {"property": "elongation", "value": 14.4, "source_context": _UNIT_TEST_CTX,
                     "unit_info": {"source_column": "TE [%]", "reported_unit": "%", "unit_verification_status": "verified_from_table_header"},
                     "elongation_type": "total"}

# TEST G — same material/processing/test condition, one uniform one total:
# must NOT be merged as replicates of a single "elongation" property.
_obs_g = e3.build_valid_observation([_elong_uniform_item, _elong_total_item], "confirmed", "confirmed_explicit")
check("TEST G: uniform and total elongation occupy two separate property slots, not one merged 'elongation'",
      set(_obs_g["properties"].keys()) == {"elongation_uniform", "elongation_total"},
      detail=str(sorted(_obs_g["properties"].keys())))
check("TEST G: neither slot reports itself as a replicate (n_replicates_reported is None for both -- each is a single, distinct value)",
      _obs_g["properties"]["elongation_uniform"]["replicate_info"]["n_replicates_reported"] is None
      and _obs_g["properties"]["elongation_total"]["replicate_info"]["n_replicates_reported"] is None)
check("TEST G: uniform value (8.8) and total value (14.4) are both preserved distinctly, neither lost/averaged",
      _obs_g["properties"]["elongation_uniform"]["value"] == 8.8 and _obs_g["properties"]["elongation_total"]["value"] == 14.4)

# TEST H — true replicate: two total-elongation measurements, same
# material/processing/test condition, same definition -> existing
# replicate policy (merge into one slot, raw values preserved) unchanged.
_elong_total_rep1 = {"property": "elongation", "value": 14.4, "source_context": _UNIT_TEST_CTX,
                     "unit_info": {"source_column": "TE [%]", "reported_unit": "%", "unit_verification_status": "verified_from_table_header"},
                     "elongation_type": "total"}
_elong_total_rep2 = {"property": "elongation", "value": 14.9, "source_context": _UNIT_TEST_CTX,
                     "unit_info": {"source_column": "TE [%]", "reported_unit": "%", "unit_verification_status": "verified_from_table_header"},
                     "elongation_type": "total"}
_obs_h = e3.build_valid_observation([_elong_total_rep1, _elong_total_rep2], "confirmed", "confirmed_explicit")
check("TEST H: two true total-elongation replicates still merge into ONE 'elongation_total' slot",
      list(_obs_h["properties"].keys()) == ["elongation_total"], detail=str(list(_obs_h["properties"].keys())))
check("TEST H: replicate policy unchanged -- n_replicates_reported=2, raw values preserved, not averaged",
      _obs_h["properties"]["elongation_total"]["replicate_info"]["n_replicates_reported"] == 2
      and _obs_h["properties"]["elongation_total"]["replicate_info"]["raw_replicate_values"] == [14.4, 14.9]
      and _obs_h["properties"]["elongation_total"]["value"] is None)

# sanity: UTS/YS grouping is completely unaffected by the elongation_type
# key (they carry no elongation_type field at all, key stays the bare name)
_uts_only_obs = e3.build_valid_observation([_rep_item_1], "confirmed", "confirmed_explicit")
check("sanity: a UTS-only observation still keys its property as plain 'UTS' (no elongation_type interference)",
      list(_uts_only_obs["properties"].keys()) == ["UTS"])

# ─────────────────────────────────────────────────────────────────────────
# Property scope / loading mode: TEST I (a bulk-tensile YS and a
# compressive YS in the SAME cluster must NEVER be merged as replicates of
# one another -- they are physically different quantities) and TEST J
# (real Acta pillar-table wording, via classify_property_scope /
# classify_loading_mode directly, mirroring TEST B-F's elongation pattern).
# ─────────────────────────────────────────────────────────────────────────

_ys_bulk_item = {"property": "YS", "value": 530.0, "source_context": _UNIT_TEST_CTX,
                  "unit_info": {"source_column": "YS [MPa]", "reported_unit": "MPa",
                                "unit_verification_status": "verified_from_table_header"},
                  "property_scope": "unspecified", "loading_mode": "unspecified"}
_ys_compressive_item = {"property": "YS", "value": 277.0, "source_context": _UNIT_TEST_CTX,
                          "unit_info": {"source_column": "Compressive yield strength /[MPa]", "reported_unit": "MPa",
                                        "unit_verification_status": "verified_from_table_header"},
                          "property_scope": "unspecified", "loading_mode": "compressive"}

# TEST I — mixed bulk + compressive YS in one cluster
_obs_i = e3.build_valid_observation([_ys_bulk_item, _ys_compressive_item], "confirmed", "confirmed_explicit")
check("TEST I: bulk YS and compressive YS occupy two separate property slots, never merged as replicates",
      set(_obs_i["properties"].keys()) == {"YS", "YS_unspecified_compressive"},
      detail=str(sorted(_obs_i["properties"].keys())))
check("TEST I: the bulk slot keeps the plain 'YS' key with its own value untouched (530.0)",
      _obs_i["properties"]["YS"]["value"] == 530.0
      and _obs_i["properties"]["YS"]["replicate_info"]["n_replicates_reported"] is None)
check("TEST I: the compressive slot is a separate, distinctly-keyed value (277.0), not averaged into 'YS'",
      _obs_i["properties"]["YS_unspecified_compressive"]["value"] == 277.0
      and _obs_i["properties"]["YS_unspecified_compressive"]["replicate_info"]["n_replicates_reported"] is None)
check("TEST I: the compressive slot carries its own property_scope/loading_mode metadata",
      _obs_i["properties"]["YS_unspecified_compressive"]["property_scope"] == "unspecified"
      and _obs_i["properties"]["YS_unspecified_compressive"]["loading_mode"] == "compressive")

# TEST J — classify_property_scope / classify_loading_mode, Stage 3's own
# self-contained copy (mirrors classify_elongation_type's duplication
# convention), exercised against real Acta pillar-table wording.
check("TEST J: Stage 3's classify_loading_mode('Compressive yield strength /[MPa]') -> compressive",
      e3.classify_loading_mode("Compressive yield strength /[MPa]") == "compressive")
check("TEST J: Stage 3's classify_property_scope('Micropillar compressive yield strength [MPa]') -> microscale_or_local",
      e3.classify_property_scope("Micropillar compressive yield strength [MPa]") == "microscale_or_local")
check("TEST J: Stage 3's classify_property_scope('Ferrite yield strength [MPa]') -> phase_level",
      e3.classify_property_scope("Ferrite yield strength [MPa]") == "phase_level")
check("TEST J: Stage 3's classify_property_scope('YS [MPa]') -> unspecified (no cue, never defaulted to bulk)",
      e3.classify_property_scope("YS [MPa]") == "unspecified")
check("TEST J: Stage 3's classify_loading_mode('YS [MPa]') -> unspecified (no cue, never defaulted to tensile)",
      e3.classify_loading_mode("YS [MPa]") == "unspecified")


# ─────────────────────────────────────────────────────────────────────────
# ROW-LABEL MATERIAL IDENTIFIER FIX — build_source_context's deterministic
# material_identifier derivation gains a third, conservative check (after
# the existing header-keyword and caption-substring checks both fail):
# the row's own first-cell label, matched via an EXACT normalized
# comparison against the material's own grade_label -- never substring
# matching, never guessed from an ambiguous/blank/multi-material row. Root
# cause: Table 3 in the real Acta paper has an empty header on its label
# column AND a caption that never names "DP1"/"DP2" literally, so both
# EXISTING checks failed despite the row itself explicitly saying "DP1"/
# "DP2" -- a logic omission, not missing metadata.
# ─────────────────────────────────────────────────────────────────────────

ROW_LABEL_PAPER = {
    "doi": "10.9999/row-label-test",
    "tables": [
        {"caption": "Mechanical properties comparison for both grades.",  # deliberately never names DP1/DP2
         "headers": ["", "YS (MPa)", "UTS (MPa)"],  # empty label-column header, matching the real Acta Table 3 shape
         "rows": [
             ["DP1", "530", "830"],          # row 0: DP1's own row
             ["DP2", "494", "830"],          # row 1: DP2's own row -- SAME UTS=830 as row 0, deliberately
             ["", "450", "800"],             # row 2: blank row label
             ["DP1 / DP2", "999", "777"],    # row 3: multi-material row
             ["S1", "500", "800"],           # row 4: specimen-style label, not a grade match
         ]},
        {"caption": "Properties table with an explicit Grade column",
         "headers": ["Grade", "YS (MPa)"], "rows": [["DP1", "530"]]},
    ],
    "sections": [],
}


def _row_sc(row_index, grade_label):
    return e3.build_source_context(
        ROW_LABEL_PAPER, {"type": "table", "table_index": 0, "row_index": row_index, "column": "YS (MPa)"},
        material_grade_label=grade_label, processing_state_type="baseline_or_as_received_state",
        group_has_test_condition_evidence=False)


# 1. exact row-label match populates material_identifier
check("row-label 1: row 0 ('DP1') + material grade_label='DP1' -> material_identifier='DP1'",
      _row_sc(0, "DP1")["material_identifier"] == "DP1", detail=str(_row_sc(0, "DP1")))
check("row-label 1b: row 1 ('DP2') + material grade_label='DP2' -> material_identifier='DP2'",
      _row_sc(1, "DP2")["material_identifier"] == "DP2", detail=str(_row_sc(1, "DP2")))

# 2. explicit conflicting row label does NOT populate it
check("row-label 2: row 0 ('DP1') + material grade_label='DP2' -> material_identifier stays None (conflict)",
      _row_sc(0, "DP2")["material_identifier"] is None, detail=str(_row_sc(0, "DP2")))
check("row-label 2b: row 1 ('DP2') + material grade_label='DP1' -> material_identifier stays None (conflict)",
      _row_sc(1, "DP1")["material_identifier"] is None, detail=str(_row_sc(1, "DP1")))

# 3. blank row label stays None
check("row-label 3: blank row label (row 2) -> material_identifier stays None",
      _row_sc(2, "DP1")["material_identifier"] is None and _row_sc(2, "DP2")["material_identifier"] is None)

# 4. multi-material row stays None -- no substring matching, even though "DP1" IS a substring of "DP1 / DP2"
check("row-label 4: multi-material row ('DP1 / DP2') -> material_identifier stays None for DP1 "
      "(no substring matching)",
      _row_sc(3, "DP1")["material_identifier"] is None, detail=str(_row_sc(3, "DP1")))
check("row-label 4b: multi-material row ('DP1 / DP2') -> material_identifier stays None for DP2 either",
      _row_sc(3, "DP2")["material_identifier"] is None, detail=str(_row_sc(3, "DP2")))

# 5. existing caption-based path unchanged -- re-affirms the pre-existing
# SYN_PAPER check above (sc["material_identifier"] == "DP780") is untouched;
# that check already re-runs as part of this same suite and still passes.
check("row-label 5: pre-existing caption-based path (SYN_PAPER, 'DP780') is unaffected by this fix",
      sc["material_identifier"] == "DP780", detail=str(sc))

# 6. existing header-keyword path unchanged -- an explicit "Grade" header
# still resolves via the FIRST (header-keyword) check, never even reaching
# the new row-label fallback.
sc_header = e3.build_source_context(
    ROW_LABEL_PAPER, {"type": "table", "table_index": 1, "row_index": 0, "column": "YS (MPa)"},
    material_grade_label="DP1", processing_state_type="baseline_or_as_received_state",
    group_has_test_condition_evidence=False)
check("row-label 6: explicit 'Grade' header column still resolves via the existing header-keyword path",
      sc_header["material_identifier"] == "DP1", detail=str(sc_header))

# 7. identical numeric property values across two rows (UTS=830 in both row 0
# and row 1) must have no effect on identity -- already implicit in tests 1/1b
# above (both correctly resolve to their OWN grade despite the coincidence),
# restated explicitly here for direct traceability to the required case.
check("row-label 7: identical UTS=830 in both rows does not affect row-label identity resolution "
      "(row 0 -> DP1, row 1 -> DP2, independently of the shared value)",
      _row_sc(0, "DP1")["material_identifier"] == "DP1" and _row_sc(1, "DP2")["material_identifier"] == "DP2")

# 8. specimen-style row label that does not equal the material grade label
# does not confirm
check("row-label 8: specimen-style row label ('S1') does not confirm material_identifier for grade_label='DP1'",
      _row_sc(4, "DP1")["material_identifier"] is None, detail=str(_row_sc(4, "DP1")))

# 9. end-to-end: table-row-derived material_identifier flows through the
# UNCHANGED resolve_material_identity gate and correctly reaches "confirmed"
# -- the gate itself needed no modification.
_row_item_dp1 = {"property": "YS", "value": 530.0, "source_context": _row_sc(0, "DP1")}
_confirmed = e3.resolve_material_identity([_row_item_dp1], "DP1", n_materials_in_paper=2)
check("row-label 9 (end-to-end): a single-item cluster with a row-label-derived material_identifier "
      "resolves 'confirmed' via the unmodified resolve_material_identity gate",
      _confirmed == "confirmed", detail=str(_confirmed))
_row_item_conflict = {"property": "YS", "value": 494.0, "source_context": _row_sc(1, "DP1")}
_unconfirmed = e3.resolve_material_identity([_row_item_conflict], "DP1", n_materials_in_paper=2)
check("row-label 9b (end-to-end): a conflicting-row item (no identifier set) stays 'unresolved' "
      "through the same unmodified gate (n_materials_in_paper=2, so no single-material fallback applies)",
      _unconfirmed == "unresolved", detail=str(_unconfirmed))


# ─────────────────────────────────────────────────────────────────────────
# REAL ACTA OFFLINE REPLAY (test 10 + required replay section) — no LLM
# call anywhere. Rebuilds source_context directly against the real,
# current local processed Acta JSON for Table 3's property rows AND
# confirms Table 0's existing composition/caption-based path is completely
# unchanged.
# ─────────────────────────────────────────────────────────────────────────

_ACTA_STAGE3_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "10-1016_j-actamat-2019-11-002.json"
if _ACTA_STAGE3_PATH.exists():
    _acta_paper = json.loads(_ACTA_STAGE3_PATH.read_text())

    def _acta_t3_sc(row_index, grade_label):
        return e3.build_source_context(
            _acta_paper, {"type": "table", "table_index": 3, "row_index": row_index, "column": "YS [MPa]"},
            material_grade_label=grade_label, processing_state_type="baseline_or_as_received_state",
            group_has_test_condition_evidence=False)

    _dp1_r0 = _acta_t3_sc(0, "DP1")
    _dp1_r1 = _acta_t3_sc(1, "DP1")
    _dp2_r1 = _acta_t3_sc(1, "DP2")
    _dp2_r0 = _acta_t3_sc(0, "DP2")

    check("REAL ACTA REPLAY (Table 3): DP1 material + T3:R0 -> material_identifier == 'DP1'",
          _dp1_r0["material_identifier"] == "DP1", detail=str(_dp1_r0))
    check("REAL ACTA REPLAY (Table 3): DP1 material + T3:R1 -> material_identifier is NOT 'DP1'",
          _dp1_r1["material_identifier"] != "DP1", detail=str(_dp1_r1))
    check("REAL ACTA REPLAY (Table 3): DP2 material + T3:R1 -> material_identifier == 'DP2'",
          _dp2_r1["material_identifier"] == "DP2", detail=str(_dp2_r1))
    check("REAL ACTA REPLAY (Table 3): DP2 material + T3:R0 -> material_identifier is NOT 'DP2'",
          _dp2_r0["material_identifier"] != "DP2", detail=str(_dp2_r0))

    # pass the correctly-matched real source contexts through the
    # UNMODIFIED resolve_material_identity gate
    _acta_dp1_item = {"property": "YS", "value": 530.0, "source_context": _dp1_r0}
    _acta_dp1_confirmed = e3.resolve_material_identity([_acta_dp1_item], "DP1", n_materials_in_paper=3)
    check("REAL ACTA REPLAY: DP1's T3:R0 item resolves material_identity='confirmed' via the unmodified gate",
          _acta_dp1_confirmed == "confirmed", detail=str(_acta_dp1_confirmed))
    _acta_dp2_item = {"property": "YS", "value": 494.0, "source_context": _dp2_r1}
    _acta_dp2_confirmed = e3.resolve_material_identity([_acta_dp2_item], "DP2", n_materials_in_paper=3)
    check("REAL ACTA REPLAY: DP2's T3:R1 item resolves material_identity='confirmed' via the unmodified gate",
          _acta_dp2_confirmed == "confirmed", detail=str(_acta_dp2_confirmed))

    # test 10 — Table 0 (composition) existing caption-based behaviour is unchanged
    _acta_t0_sc = e3.build_source_context(
        _acta_paper, {"type": "table", "table_index": 0, "row_index": 0, "column": "C"},
        material_grade_label="DP1", processing_state_type="baseline_or_as_received_state",
        group_has_test_condition_evidence=False)
    check("row-label 10 / REAL ACTA REPLAY: Table 0 (composition) still resolves 'DP1' via the "
          "existing caption-substring path, unaffected by the new row-label fallback",
          _acta_t0_sc["material_identifier"] == "DP1", detail=str(_acta_t0_sc))
else:
    check("REAL ACTA REPLAY: real Acta paper file present for offline replay", False, detail=str(_ACTA_STAGE3_PATH))


# ─────────────────────────────────────────────────────────────────────────
# ROW-ORIENTED CANDIDATE UNIT CARRY-THROUGH — Stage 2's row-oriented
# fallback (evidence_prep_v2.py) supplies a NEW pe field, "unit_source_
# text", set to the row's own label (e.g. "UTS (MPa)") for row-oriented
# candidates only -- absent (None) for every existing column-oriented
# candidate. The ONLY Stage 3 change is a one-line fallback in
# process_evidence_group's table-sourced item loop: prefer pe.get(
# "unit_source_text") when present, else fall back to loc.get("column")
# exactly as before. This is required because location["column"] must stay
# the REAL column header ("AB") for reverify_property_item/
# resolve_column_index to keep working unmodified -- it carries no unit,
# so the unit signal must come from somewhere else.
# ─────────────────────────────────────────────────────────────────────────

ROW_ORIENTED_UNIT_PAPER = {
    "doi": "10.9999/row-oriented-unit-test",
    "tables": [{"caption": "Mechanical properties by condition", "headers": ["Property", "AB", "A"],
                "rows": [["UTS (MPa)", "1100", "1800"]]}],
    "sections": [],
}
ROW_ORIENTED_UNIT_MATERIAL = {"material_id": "M1", "grade_label": "DP1", "composition_evidence": [], "evidence_groups": []}
ROW_ORIENTED_UNIT_GROUP = {
    "group_id": "G1", "processing_state_type": "baseline_or_as_received_state",
    "processing_evidence": [], "test_condition_evidence": [],
    "property_evidence": [{"property": "UTS",
                            "location": {"type": "table", "table_index": 0, "row_index": 0, "column": "AB"},
                            "quote": "1100", "elongation_type": None, "property_scope": "unspecified",
                            "loading_mode": "unspecified", "unit_source_text": "UTS (MPa)"}],
}
result_unit = e3.process_evidence_group(ROW_ORIENTED_UNIT_PAPER, ROW_ORIENTED_UNIT_MATERIAL, ROW_ORIENTED_UNIT_GROUP,
                                         n_materials_in_paper=1, llm_enabled=False)
_unit_props = (result_unit["valid_observations"][0]["properties"] if result_unit["valid_observations"]
               else {p["property"]: p for p in result_unit["orphan_property_evidence"][0]["properties"]}
               if result_unit["orphan_property_evidence"] else {})
check("unit carry-through: row-oriented candidate's unit is correctly detected as 'MPa' from "
      "unit_source_text ('UTS (MPa)'), even though location.column ('AB') itself has no unit",
      _unit_props.get("UTS", {}).get("reported_unit") == "MPa", detail=str(_unit_props))
check("unit carry-through: unit_verification_status is verified_from_table_header (same status the "
      "ordinary column-oriented path already uses -- no new status introduced)",
      _unit_props.get("UTS", {}).get("unit_verification_status") == "verified_from_table_header", detail=str(_unit_props))
check("unit carry-through: the numeric value itself (1100) is still correctly re-verified/parsed unchanged",
      _unit_props.get("UTS", {}).get("value") == 1100.0, detail=str(_unit_props))

# regression: an ORDINARY column-oriented item (no unit_source_text key at
# all, exactly like every pre-existing property_evidence item) is unaffected
ORDINARY_UNIT_GROUP = {
    "group_id": "G1", "processing_state_type": "baseline_or_as_received_state",
    "processing_evidence": [], "test_condition_evidence": [],
    "property_evidence": [{"property": "UTS",
                            "location": {"type": "table", "table_index": 0, "row_index": 0, "column": "UTS (MPa)"},
                            "quote": "900"}],
}
ORDINARY_UNIT_PAPER = {
    "doi": "10.9999/ordinary-unit-test",
    "tables": [{"caption": "test", "headers": ["Grade", "UTS (MPa)"], "rows": [["DP1", "900"]]}],
    "sections": [],
}
result_ordinary = e3.process_evidence_group(ORDINARY_UNIT_PAPER, ROW_ORIENTED_UNIT_MATERIAL, ORDINARY_UNIT_GROUP,
                                             n_materials_in_paper=1, llm_enabled=False)
_ordinary_props = (result_ordinary["valid_observations"][0]["properties"] if result_ordinary["valid_observations"]
                    else {p["property"]: p for p in result_ordinary["orphan_property_evidence"][0]["properties"]}
                    if result_ordinary["orphan_property_evidence"] else {})
check("unit carry-through regression: an ordinary column-oriented item (no unit_source_text key present at "
      "all) still derives its unit from location.column exactly as before this patch",
      _ordinary_props.get("UTS", {}).get("reported_unit") == "MPa", detail=str(_ordinary_props))


print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
