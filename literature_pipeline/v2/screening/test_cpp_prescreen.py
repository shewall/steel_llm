"""
test_cpp_prescreen.py — offline, dependency-free sanity tests for Stage 1
screening's deterministic decision-rule logic (apply_llm_decision_rules).
No LLM calls, no network access anywhere below — apply_llm_decision_rules is
a pure function of an already-normalized llm-output dict and a
confirmed_variables set, tested directly.

Run: python literature_pipeline/v2/screening/test_cpp_prescreen.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from literature_pipeline.v2.screening import cpp_prescreen as s1

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not condition:
        FAILURES.append(label)


# ─────────────────────────────────────────────────────────────────────────
# Fixture helper — a fully-populated, already-normalized llm-output dict
# (the shape normalize_llm_output() produces), with sensible "clean,
# clearly relevant" defaults. Individual tests override only the fields
# they care about.
# ─────────────────────────────────────────────────────────────────────────

def _llm(**overrides) -> dict:
    base = {
        "material_system": "DP steel",
        "material_system_coherence": "single_material",
        "variables_changed": ["none_single_condition"],
        "heat_treatment_processing_support": "not_claimed",
        "heat_treatment_processing_support_text": "",
        "study_nature": "original_experimental",
        "composition_status": "complete",
        "processing_status": "detailed_processing",
        "properties_reported": {"UTS": True, "YS": True, "elongation": True},
        "secondary_domain_tags": [],
        "cpp_content_is_incidental": False,
        "evidence_modality": ["table"],
        "evidence_sufficiency": "high",
        "notes": "",
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────
# S4 — sanity: existing clean cpp_relevant cases remain unchanged (baseline,
# run first so a later regression in the shared step-7 branch would also be
# caught here for a completely unrelated, non-test-only case)
# ─────────────────────────────────────────────────────────────────────────

_s4_llm = _llm()
_s4_confirmed = {"none_single_condition"}
_s4_relevance, _s4_reason, _s4_rule = s1.apply_llm_decision_rules(_s4_llm, _s4_confirmed, det_has_property_header_table=True)
check("S4: clean single-condition, composition+processing OK, property present -> cpp_relevant",
      _s4_relevance == "cpp_relevant" and _s4_reason == "keep_cpp_relevant",
      detail=str((_s4_relevance, _s4_reason, _s4_rule)))

# ─────────────────────────────────────────────────────────────────────────
# S3 — existing step-8 composition-not-observed behaviour (the GENERAL,
# non-test-only case) remains unchanged by this fix
# ─────────────────────────────────────────────────────────────────────────

_s3_llm = _llm(composition_status="not_observed", variables_changed=["heat_treatment_processing"],
               heat_treatment_processing_support="explicit",
               heat_treatment_processing_support_text="samples were annealed at 850C")
_s3_confirmed = {"heat_treatment_processing"}  # pre-confirmed directly (bypassing compute_confirmed_variables)
_s3_relevance, _s3_reason, _s3_rule = s1.apply_llm_decision_rules(_s3_llm, _s3_confirmed, det_has_property_header_table=True)
check("S3: step-8 general composition_not_observed case still -> ambiguous/composition_not_observed_otherwise_relevant",
      _s3_relevance == "ambiguous_relevance" and _s3_reason == "composition_not_observed_otherwise_relevant"
      and _s3_rule == "llm_composition_not_observed",
      detail=str((_s3_relevance, _s3_reason, _s3_rule)))

# ─────────────────────────────────────────────────────────────────────────
# S1 — THE FIX: test-condition-only variables + composition_status ==
# not_observed + otherwise-relevant evidence (property present, processing
# OK). Pre-fix: falls straight to the test_only branch's hard non_cpp
# return. Post-fix: deferred to ambiguous_relevance via the existing
# composition_not_observed_otherwise_relevant reason code.
# ─────────────────────────────────────────────────────────────────────────

_s1_llm = _llm(composition_status="not_observed", processing_status="detailed_processing")
_s1_confirmed = {"test_temperature"}  # a pure test-condition-only variable
_s1_relevance, _s1_reason, _s1_rule = s1.apply_llm_decision_rules(_s1_llm, _s1_confirmed, det_has_property_header_table=True)
check("S1: test-only variable + composition not_observed + otherwise relevant -> ambiguous_relevance (was: hard non_cpp)",
      _s1_relevance == "ambiguous_relevance", detail=str((_s1_relevance, _s1_reason, _s1_rule)))
check("S1: reuses the EXISTING composition_not_observed_otherwise_relevant reason code (no new reason code invented)",
      _s1_reason == "composition_not_observed_otherwise_relevant", detail=str(_s1_reason))

# variant: processing_status == "unclear" (not just PROCESSING_OK) — the
# test-only branch's own PRE-EXISTING ambiguous condition already accepts
# "unclear" processing (see rule_fired=llm_nonstandard_variable_partial
# above), so the new composition-not-observed path should be consistent
# with that and also accept "unclear" processing, not just PROCESSING_OK.
_s1b_llm = _llm(composition_status="not_observed", processing_status="unclear")
_s1b_relevance, _s1b_reason, _s1b_rule = s1.apply_llm_decision_rules(_s1b_llm, _s1_confirmed, det_has_property_header_table=True)
check("S1 variant: test-only + composition not_observed + processing unclear -> still ambiguous (consistent with the branch's existing unclear-processing tolerance)",
      _s1b_relevance == "ambiguous_relevance", detail=str((_s1b_relevance, _s1b_reason, _s1b_rule)))

# variant: no property reported at all -- weaker "otherwise relevant"
# signal; this fix requires any_property, mirroring step 8's own gate, so
# this should NOT be caught by the new path and falls through to the
# unchanged hard exclusion.
_s1c_llm = _llm(composition_status="not_observed", properties_reported={"UTS": False, "YS": False, "elongation": False})
_s1c_relevance, _s1c_reason, _s1c_rule = s1.apply_llm_decision_rules(_s1c_llm, _s1_confirmed, det_has_property_header_table=True)
check("S1 variant: test-only + composition not_observed + NO property reported -> still hard non_cpp (any_property gate preserved)",
      _s1c_relevance == "non_cpp" and _s1c_reason == "nonstandard_independent_variable_excluded",
      detail=str((_s1c_relevance, _s1c_reason, _s1c_rule)))

# ─────────────────────────────────────────────────────────────────────────
# S2 — the SAME test-only pattern, but composition is "unclear" rather than
# "not_observed". At the time this test was originally written (the
# composition-not-observed-only fix), "unclear" was assumed to be a
# stronger, positively-not-confirmed signal and this test asserted hard
# exclusion must remain. A SEPARATE, subsequent audit examined this
# assumption directly against the prompt/schema and found it unsupported:
# "unclear" has no field-specific stronger-negative definition in the
# prompt, and is also normalize_enum()'s fallback for a malformed/missing
# raw value. The accepted verdict of that audit ("CHANGE UNCLEAR TO
# DEFER") supersedes this test's original assertion -- see U1-U8 below for
# the full, dedicated test coverage of the new behaviour. This test is
# updated (not deleted) to keep the historical assertion traceable and to
# continue guarding against a future regression back to hard exclusion.
# ─────────────────────────────────────────────────────────────────────────

_s2_llm = _llm(composition_status="unclear", processing_status="detailed_processing")
_s2_relevance, _s2_reason, _s2_rule = s1.apply_llm_decision_rules(_s2_llm, _s1_confirmed, det_has_property_header_table=True)
check("S2 (superseded by the composition-unclear-deferral fix, see U1): "
      "test-only + composition_status=unclear -> now DEFERS (ambiguous_relevance/composition_unclear_otherwise_relevant), no longer hard-excluded",
      _s2_relevance == "ambiguous_relevance" and _s2_reason == "composition_unclear_otherwise_relevant"
      and _s2_rule == "llm_test_only_composition_unclear",
      detail=str((_s2_relevance, _s2_reason, _s2_rule)))

# S2 also confirms the pre-existing partial-ambiguous path (composition
# already OK) is completely untouched by this change
_s2b_llm = _llm(composition_status="partial", processing_status="unclear")
_s2b_relevance, _s2b_reason, _s2b_rule = s1.apply_llm_decision_rules(_s2b_llm, _s1_confirmed, det_has_property_header_table=True)
check("S2 sanity: test-only + composition=partial (already COMPOSITION_OK) -> unchanged ambiguous/nonstandard_independent_variable_partial",
      _s2b_relevance == "ambiguous_relevance" and _s2b_reason == "nonstandard_independent_variable_partial",
      detail=str((_s2b_relevance, _s2b_reason, _s2b_rule)))


# ─────────────────────────────────────────────────────────────────────────
# composition_status == "unclear" deferral fix (U1-U8). "unclear" has no
# field-specific stronger-negative meaning in the prompt (only
# "not_observed" is explicitly defined there), and is ALSO normalize_enum's
# fallback for a malformed/missing raw value -- a parsing artifact carrying
# no information about the paper. A whole-record LLM call/parse failure
# already defers to ambiguous_relevance elsewhere in this module; a single
# field falling back the same way should not be treated more harshly than
# "not_observed". This section confirms both branches treat "unclear"
# consistently with "not_observed" while keeping the two source states
# distinguishable via distinct reason_code/rule_fired values, and never
# rewriting composition_status itself.
# ─────────────────────────────────────────────────────────────────────────

# U1 — test-condition-only branch: composition_status="unclear" + property
# present + processing OK. Pre-fix: falls to the test_only branch's hard
# non_cpp return (identical to S2 above). Post-fix: deferred.
_u1_llm = _llm(composition_status="unclear", processing_status="detailed_processing")
_u1_confirmed = {"test_temperature"}
_u1_relevance, _u1_reason, _u1_rule = s1.apply_llm_decision_rules(_u1_llm, _u1_confirmed, det_has_property_header_table=True)
check("U1: test-only + composition unclear + otherwise relevant -> ambiguous_relevance (was: hard non_cpp)",
      _u1_relevance == "ambiguous_relevance", detail=str((_u1_relevance, _u1_reason, _u1_rule)))
check("U1: uses a DISTINCT reason code from the not_observed path (composition_unclear_otherwise_relevant, not composition_not_observed_otherwise_relevant)",
      _u1_reason == "composition_unclear_otherwise_relevant", detail=str(_u1_reason))
check("U1: rule_fired is distinct from the not_observed test-only path (llm_test_only_composition_unclear)",
      _u1_rule == "llm_test_only_composition_unclear", detail=str(_u1_rule))

# U2 — general (non-test-only) branch: composition_status="unclear" +
# otherwise-relevant evidence. Pre-fix, this ALREADY resolves to
# ambiguous_relevance (via the generic step-10 catch-all,
# "unclassified_needs_review") rather than a hard exclusion -- there is no
# rule anywhere that hard-excludes on composition_status alone outside the
# test-only branch. The fix's value here is audit-trail PRECISION: a
# specific, accurate reason code instead of the generic catch-all, exactly
# mirroring step 8's existing not_observed handling.
_u2_llm = _llm(composition_status="unclear", processing_status="detailed_processing")
_u2_confirmed = {"heat_treatment_processing"}  # not a test-condition-only variable
_u2_relevance, _u2_reason, _u2_rule = s1.apply_llm_decision_rules(_u2_llm, _u2_confirmed, det_has_property_header_table=True)
check("U2: general otherwise-relevant case + composition unclear -> ambiguous_relevance",
      _u2_relevance == "ambiguous_relevance", detail=str((_u2_relevance, _u2_reason, _u2_rule)))
check("U2: reason code is the specific composition_unclear_otherwise_relevant, not the generic unclassified_needs_review catch-all",
      _u2_reason == "composition_unclear_otherwise_relevant", detail=str(_u2_reason))
check("U2: rule_fired is llm_composition_unclear (distinct from the general not_observed path's llm_composition_not_observed)",
      _u2_rule == "llm_composition_unclear", detail=str(_u2_rule))

# U3 — the same two cases with "not_observed" instead of "unclear": the
# EXISTING, already-accepted defer behaviour and its reason codes/rule_fired
# values must remain byte-for-byte unchanged by this fix.
_u3a_llm = _llm(composition_status="not_observed", processing_status="detailed_processing")
_u3a_relevance, _u3a_reason, _u3a_rule = s1.apply_llm_decision_rules(_u3a_llm, _u1_confirmed, det_has_property_header_table=True)
check("U3a: test-only + composition not_observed -> unchanged (ambiguous_relevance/composition_not_observed_otherwise_relevant/llm_test_only_composition_not_observed)",
      (_u3a_relevance, _u3a_reason, _u3a_rule) == ("ambiguous_relevance", "composition_not_observed_otherwise_relevant", "llm_test_only_composition_not_observed"),
      detail=str((_u3a_relevance, _u3a_reason, _u3a_rule)))
_u3b_llm = _llm(composition_status="not_observed", processing_status="detailed_processing")
_u3b_relevance, _u3b_reason, _u3b_rule = s1.apply_llm_decision_rules(_u3b_llm, _u2_confirmed, det_has_property_header_table=True)
check("U3b: general + composition not_observed -> unchanged (ambiguous_relevance/composition_not_observed_otherwise_relevant/llm_composition_not_observed)",
      (_u3b_relevance, _u3b_reason, _u3b_rule) == ("ambiguous_relevance", "composition_not_observed_otherwise_relevant", "llm_composition_not_observed"),
      detail=str((_u3b_relevance, _u3b_reason, _u3b_rule)))

# U4 — composition_status="partial" (already COMPOSITION_OK): existing
# behaviour unchanged in both branches.
_u4a_llm = _llm(composition_status="partial", processing_status="detailed_processing")
_u4a_relevance, _u4a_reason, _u4a_rule = s1.apply_llm_decision_rules(_u4a_llm, _u1_confirmed, det_has_property_header_table=True)
check("U4a: test-only + composition partial -> unchanged (ambiguous_relevance/nonstandard_independent_variable_partial)",
      (_u4a_relevance, _u4a_reason) == ("ambiguous_relevance", "nonstandard_independent_variable_partial"),
      detail=str((_u4a_relevance, _u4a_reason, _u4a_rule)))
_u4b_llm = _llm(composition_status="partial", processing_status="detailed_processing")
_u4b_confirmed = {"heat_treatment_processing"}
_u4b_relevance, _u4b_reason, _u4b_rule = s1.apply_llm_decision_rules(_u4b_llm, _u4b_confirmed, det_has_property_header_table=True)
check("U4b: general + composition partial -> unchanged (cpp_relevant/keep_cpp_relevant)",
      (_u4b_relevance, _u4b_reason) == ("cpp_relevant", "keep_cpp_relevant"),
      detail=str((_u4b_relevance, _u4b_reason, _u4b_rule)))

# U5 — composition_status="complete": existing behaviour unchanged.
_u5_llm = _llm(composition_status="complete", processing_status="detailed_processing")
_u5_confirmed = {"heat_treatment_processing"}
_u5_relevance, _u5_reason, _u5_rule = s1.apply_llm_decision_rules(_u5_llm, _u5_confirmed, det_has_property_header_table=True)
check("U5: general + composition complete -> unchanged (cpp_relevant/keep_cpp_relevant)",
      (_u5_relevance, _u5_reason) == ("cpp_relevant", "keep_cpp_relevant"),
      detail=str((_u5_relevance, _u5_reason, _u5_rule)))

# U6 — a genuine affirmative non_cpp case (simulation_only) is completely
# unaffected by composition_status, including "unclear" -- steps 2-6 all
# fire before the composition-status checks and are untouched by this fix.
_u6_llm = _llm(study_nature="simulation_only", composition_status="unclear")
_u6_relevance, _u6_reason, _u6_rule = s1.apply_llm_decision_rules(_u6_llm, _u1_confirmed, det_has_property_header_table=True)
check("U6: affirmative non_cpp exclusion (simulation_only) remains non_cpp regardless of composition_status",
      (_u6_relevance, _u6_reason, _u6_rule) == ("non_cpp", "simulation_only", "llm_simulation_only"),
      detail=str((_u6_relevance, _u6_reason, _u6_rule)))

# U7 — malformed/missing composition_status normalized to "unclear" by
# normalize_enum() must not become hard non_cpp solely because of that
# fallback. Exercises normalize_llm_output directly (the real path a raw
# LLM response with a missing/invalid composition_status field takes) then
# feeds the result through apply_llm_decision_rules, end-to-end.
_u7_raw = {
    "material_system": "DP steel", "material_system_coherence": "single_material",
    "variables_changed": ["test_temperature"], "heat_treatment_processing_support": "not_claimed",
    "heat_treatment_processing_support_text": "", "study_nature": "original_experimental",
    "composition_status": "this-was-not-a-valid-enum-value",  # malformed raw output
    "processing_status": "detailed_processing",
    "properties_reported": {"UTS": True, "YS": False, "elongation": False},
    "secondary_domain_tags": [], "cpp_content_is_incidental": False,
    "evidence_modality": ["table"], "evidence_sufficiency": "high", "notes": "",
}
_u7_normalized = s1.normalize_llm_output(_u7_raw)
check("U7: normalize_llm_output() falls back malformed composition_status to 'unclear' (confirms the fallback path itself, unchanged)",
      _u7_normalized["composition_status"] == "unclear", detail=str(_u7_normalized["composition_status"]))
_u7_confirmed = {"test_temperature"}
_u7_relevance, _u7_reason, _u7_rule = s1.apply_llm_decision_rules(_u7_normalized, _u7_confirmed, det_has_property_header_table=True)
check("U7: a malformed-composition-status-fallback paper is NOT hard-excluded solely because of that fallback",
      _u7_relevance == "ambiguous_relevance", detail=str((_u7_relevance, _u7_reason, _u7_rule)))

# U8 — the routing layer (run_one_paper.py, fixed in the prior task) still
# retains ambiguous+extractable cases after this change -- this fix only
# touches reason_code/relevance derivation, never extractability, so the
# routing interaction must be completely unaffected.
from literature_pipeline.v2 import run_one_paper as _rop  # noqa: E402
from literature_pipeline.v2.screening.cpp_prescreen import combined_status as _combined_status  # noqa: E402

_u8_status = _combined_status("ambiguous_relevance", "table_extractable")
_u8_decision = _rop.stage1_routing_decision("ambiguous_relevance", "table_extractable", _u8_status)
check("U8: routing layer still retains ambiguous_relevance+table_extractable as retain_ambiguous_extractable (unaffected by this fix)",
      _u8_decision == "retain_ambiguous_extractable", detail=str(_u8_decision))


print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
