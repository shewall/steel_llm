"""
extract_observations.py — V2 Stage 3: Numerical Extraction.

Runs only on evidence_groups produced by Stage 2 (frozen,
literature_pipeline/v2/evidence/evidence_prep.py) for papers with
stage2_relevance in {cpp_relevant, ambiguous_relevance}. Stage 2 produced a
MAP — WHERE composition/processing/property evidence lives, and Stage 2's own
best guess at which pieces belong together (an "evidence_group"). Stage 2's
evidence_groups are PRIORS, not ground truth observations: Stage 3 re-derives
every value directly from data/processed/<doi>.json and independently decides
whether a cluster of evidence is actually safe to present as one CPP
observation.

── Terminology ──────────────────────────────────────────────────────────────
A Stage 3 "observation" is a material + processing-state + (optional)
test-condition combination with at least one provenance-verified property
value POSITIVELY linked to that state. This is a materially stronger claim
than a Stage 2 "evidence_group" — an evidence_group is a candidate; an
observation is a claim Stage 3 is willing to stand behind.

── The three observation_status outcomes (never a binary safe/unsafe) ───────
  valid_observation        — material identity + processing-state identity
                              positively established, AND at least one
                              property re-verified and positively linked to
                              that state.
  orphan_property_evidence — a property value passed its OWN provenance
                              re-verification, but material identity and/or
                              processing-state identity could not be
                              POSITIVELY established for it. Preserved with
                              full provenance, never silently dropped, never
                              promoted to an observation on its own.
  rejected_candidate       — the cited location does not resolve, or the
                              claimed value is not actually present there.

property_completeness (properties_present / completeness_tier) is DESCRIPTIVE
ONLY and never gates observation_status. A source that legitimately reports
only UTS+YS is not "incomplete" or "unsafe" — see design report §1.

── The observation-construction gate (frozen design, see chat history) ─────
Six explicit steps, A through F:
  A. source/provenance validity   — deterministic re-read against the actual
                                     paper JSON; Stage 2's own quote/location
                                     claim is never trusted directly.
  B. material identity            — "confirmed" requires a POSITIVE explicit
                                     identifier match at/near the evidence
                                     location. A single-material paper with NO
                                     explicit identifier anywhere gets the
                                     weaker "assumed_single_material" status,
                                     which does NOT pass the gate on its own
                                     (material identity alone must never be
                                     sufficient — see gate_construct_candidates).
  C. processing-state identity    — "confirmed_explicit"/"confirmed_baseline"
                                     require an explicit processing label, a
                                     baseline/as-received claim, a qualifying
                                     shared specimen identifier, or (for
                                     same-row merges only, and only when >1
                                     item was actually merged) the row itself.
  D. test-condition identity      — resolved per property item:
                                     confirmed_explicit / default_implicit /
                                     unresolved. default_implicit is a real,
                                     recorded state (paper genuinely reports
                                     one unvaried default) — never a silent
                                     fallback for "field absent."
  E. property-to-property linkage — positively_compatible() requires
                                     material_compatible AND
                                     processing_compatible AND
                                     test_condition_compatible. Same-row is
                                     the one auto-compatible case (the row
                                     itself is the shared identifier).
                                     ABSENCE of a contradiction is NEVER
                                     sufficient on its own to merge.
  F. emission decision            — only "confirmed" material identity AND
                                     "confirmed_explicit"/"confirmed_baseline"
                                     processing identity together yield
                                     valid_observation; everything else
                                     (including assumed_single_material) is
                                     orphan_property_evidence.

── Composition is a reference, not a merge ──────────────────────────────────
Composition lives once on a Material, keyed by material_id. Multiple
processing_conditions of the SAME positively-identified material reference it
by ID. This is architecturally different from property-to-property linkage —
it is never run through positively_compatible().

── Replicates ────────────────────────────────────────────────────────────────
Multiple evidence items for the SAME property that positively cluster
together (same material + processing + test condition) are preserved as
raw_replicate_values on that property, never averaged by this code and never
turned into separate fake processing conditions.

── Deterministic vs. LLM division of labor ──────────────────────────────────
Whenever an evidence location is a table cell, the value is read directly by
indexing into the processed paper JSON — zero LLM calls. The LLM (not yet
wired into any run path in this file — see "Not run yet" below) is reserved
for: prose-embedded values, multi-step processing-route structuring from
prose, and linkage plausibility judgments Stage 3's deterministic layer
cannot resolve (e.g. identifiers only stated in prose, several paragraphs
from the property mention).

── Bounded local discovery only ──────────────────────────────────────────────
This module's scope is Stage 2's evidence_groups plus (once the LLM step is
implemented) evidence local to the SAME excerpt Stage 2 already opened. It
never independently re-scans a paper for evidence Stage 2 never surfaced at
all — that would reopen Stage 2's scope. This is a known, accepted limitation
(see chat-history design report §16/§23), not something this file works
around.

── What this stage must NOT do ──────────────────────────────────────────────
  - No unsafe cross-condition merging (the entire point of the gate above)
  - No zero-filling missing composition elements, no derived quantities
    (Fe-balance arithmetic, carbon equivalent, etc.)
  - No silent replicate averaging
  - No plausibility QC (UTS>=YS, duplicate detection — Stage 5's job)
  - No modification of literature_pipeline/v2/screening/cpp_prescreen.py or
    literature_pipeline/v2/evidence/evidence_prep.py (both frozen), or any
    legacy pipeline file — this module is fully self-contained and does not
    import from either.

── Not run yet ───────────────────────────────────────────────────────────────
This file is implementation only, at the deterministic-gate stage. The LLM
call scaffold (call_llm / call_llm_with_retry) exists so this file compiles
end-to-end and so the infrastructure fix below can be exercised, but nothing
in this file invokes it, and no SYSTEM_PROMPT for actual extraction content
has been written or reviewed — prose-sourced extraction and linkage judgment
require a dedicated prompt-design review round (mirroring how Stage 1 and
Stage 2's prompts were each separately designed and approved), not yet done.
main() is deliberately NOT implemented in this file for that reason: there is
nothing yet for it to safely orchestrate against real papers.

── Infrastructure note: the unresolved Stage 2 705.8s timeout anomaly ──────
Investigated at the code level only (Stage 2 was not modified, and no live
calls were made to reproduce it). Stage 2's call_llm() passes a single float
`timeout=LLM_CALL_TIMEOUT_S` to requests.post(). In the `requests`/urllib3
stack (requests 2.32.5 / urllib3 2.6.2, this environment), a single-value
timeout is NOT a wall-clock deadline for the whole request — it is applied
separately to the connect phase and to EACH individual socket read
operation. If the server (Ollama, stream:false at the API level but HTTP
transport can still deliver the body across multiple TCP reads/chunks) sends
any bytes periodically, each read resets its own per-operation timer, and
total elapsed time can exceed the configured value substantially without
ever raising requests.exceptions.Timeout — a well-documented gotcha of this
timeout parameter, not a bug in the value chosen. This is a plausible,
code-level explanation for how a call configured at 240s ran 705.8s without
erroring; it was not empirically reproduced here (no LLM calls were made),
so it remains a hypothesis, not a confirmed root cause. Stage 2 itself is
left untouched, per instruction. See "Patch log" below for how Stage 3's
own equivalent mechanism was built, found broken by the baseline run, and
fixed.

── Patch log ─────────────────────────────────────────────────────────────────
v0.1 (gate-only): initial deterministic observation-construction gate,
approved and frozen at the design level before any LLM wiring existed.

v0.2 (local-context layer): added the local evidence-bundle construction,
LLM call scaffold, and hierarchical assembly. The first wall-clock-watchdog
attempt wrapped call_llm() in `with concurrent.futures.ThreadPoolExecutor()
as pool: ... future.result(timeout=N)`, intended as a genuine hard deadline.

v0.3 (baseline-run patch, after the 6-paper Stage 3 baseline): the baseline
run surfaced four confirmed defects, each fixed here with the smallest
general (non-paper-specific) change:
  1. UNSAFE CROSS-LOCATION MERGE (critical): processing_compatible() treated
     any two items sharing the group-level is_baseline_or_as_received=True
     flag as compatible, with no location-specific evidence — confirmed on
     heliyon-2023-e20960, which merged UTS from a standard-tensile table
     with YS from a table explicitly captioned as an RPS pre-strain/backing
     condition. Fixed by removing that branch from processing_compatible()
     entirely — baseline is a group-level PRIOR, never sufficient on its own
     for cross-location merging.
  2. SINGLETON BASELINE OVER-CONFIRMATION: even after fix (1), a singleton
     cluster could still resolve to processing_identity="confirmed_baseline"
     purely by inheriting the group-level flag, with nothing location-
     specific backing it — which would have let heliyon's now-separated UTS
     and YS singletons each become "confirmed_baseline" independently.
     Fixed in resolve_processing_identity(): a multi-item cluster confirms
     baseline only when its own members agree on exactly one context
     signature (table caption / section heading); a singleton confirms
     baseline only when the WHOLE group's cited evidence resolves to
     exactly one context signature (not zero — an empty signature set is
     "no evidence", not "homogeneous" — and not more than one). Purely
     structural (a signature-count comparison), no keyword matching of any
     kind, so it generalizes beyond this one paper.
  3. UNHANDLED CRASH: build_local_bundle()/_render_bundle_tables() indexed
     paper["tables"][i] before validating the location, crashing on an
     out-of-range table_index Stage 2 itself had already flagged
     location_verified=False (confirmed on dt-2022-10-010). Fixed by
     validating every location via validate_location() before use; invalid
     ones are skipped and returned in a new invalid_locations field for the
     caller to record as rejected candidates, never crashing the paper.
  4. BROKEN WALL-CLOCK WATCHDOG: confirmed via the baseline run (a call ran
     697.8s against a configured 260s cap) and an isolated reproduction that
     `with ThreadPoolExecutor() as pool: ...` blocks on shutdown(wait=True)
     regardless of future.result()'s own timeout, since a running task can't
     be cancelled. Replaced with a plain daemon thread + Thread.join(timeout=
     ...), a genuine caller-side hard deadline — see call_llm()'s own
     docstring for exactly what it does and doesn't bound.
  5. WASTED CALLS ON EMPTY BUNDLES: jmrt-2020-06-063's only cited location
     was a genuinely empty (0-paragraph) section; the LLM was still called
     with an essentially content-free prompt and returned a plausible but
     unverifiable hallucinated span (correctly caught downstream by span
     verification, but only after a ~700s wasted call). Fixed with
     bundle_has_usable_content(), checked before run_llm_extraction_pass()
     is ever invoked.
Same-table citation repair (a real recall gap surfaced by
ijmecsci-2025-110565, where Stage 2 cited the right table but the wrong
row/column) was analyzed and explicitly deferred, not implemented — it is a
recall feature, not a safety fix, and warrants its own dedicated design
round rather than being bundled into a safety patch.

v0.4 (property-unit provenance, additive only): every property record (in
both valid_observation and orphan_property_evidence) now also carries
source_column / reported_unit / unit_verification_status. Motivation: the
dataset-construction layer maps UTS/YS/elongation into columns named
*_MPa/*_pct, but had no way to independently confirm the source actually
reported those units. Table-sourced values: the header Stage 2 already
carries in location["column"] is regex-matched for a recognized unit token
(GPa/MPa/Pa/%/percent), zero LLM involvement. Prose-sourced values: the
model's existing reported_unit claim (already part of the prompt/verified-
output schema — no prompt text changed, PROMPT_VERSION unchanged) is
trusted only if independently found inside the already span-verified
source_span; an unconfirmed claim never populates reported_unit, only the
"reported_by_model_unverified" status. This patch does not read or write
any other field, does not change clustering, compatibility, or the
valid/orphan/rejected emission decision — verified by re-running the full
pre-existing test_stage3_gate.py suite unmodified alongside the new tests.
"""

import json
import re
import threading
from pathlib import Path

import requests

PROC_DIR = Path("data/processed")
EVIDENCE_DIR = Path("data/evidence_v2")  # Stage 2 output — read only
OUT_DIR = Path("data/extracted_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-next:80b-a3b-instruct-q4_K_M"

CODE_VERSION = "extract_observations_v0.8_row_oriented_unit_carry_through"
PROMPT_VERSION = "stage3_local_context_v1"  # unchanged by v0.4 — no SYSTEM_PROMPT text was modified;
                                             # prose reported_unit was already part of this prompt version's schema

LLM_CALL_TIMEOUT_S = 240
LLM_WALLCLOCK_HARD_TIMEOUT_S = 260  # slightly above the per-call timeout so a
                                     # clean requests.Timeout is preferred when
                                     # it does fire; this is the backstop for
                                     # exactly the anomaly documented above
LLM_MAX_ATTEMPTS = 2

# Bounded local discovery: a cited paragraph's "local window" is its own
# section only, +/- this many paragraphs, clipped at the section boundary.
# Never crosses into a different section, never reaches a table/section not
# already cited by the evidence_group being processed.
LOCAL_PARAGRAPH_WINDOW = 2


# ══════════════════════════════════════════════════════════════════════════
# A. SOURCE / PROVENANCE VALIDITY — deterministic, self-contained
#    (same policy as Stage 2's provenance layer, copied not imported, plus
#    numeric-equivalence recovery of exact-value formatting differences)
# ══════════════════════════════════════════════════════════════════════════

def _normalize_for_match(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower())
    s = re.sub(r"\s*([%°±])\s*", r"\1", s)
    return s.strip()


def parse_numeric_token(token) -> float | None:
    """Deterministic numeric parse — not fuzzy matching. Strips thousands
    separators and unit-adjacent symbols, then requires the remainder to be
    a plain signed decimal number."""
    if token is None:
        return None
    s = str(token).strip().replace(",", "")
    s = re.sub(r"[%°]", "", s).replace("±", "").strip()
    if not re.match(r"^-?\d+(\.\d+)?$", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def numeric_equivalent(token_a, token_b) -> bool:
    """True only if both sides parse to the SAME float after deterministic
    parsing — e.g. "765" == "765.0" == "765.00", but "765" != "766". This is
    a different operation than string/substring matching and is applied only
    to numeric fields; it recovers exact-value formatting mismatches (the
    confirmed ijmecsci decimal-notation case) without loosening anything
    semantically."""
    a, b = parse_numeric_token(token_a), parse_numeric_token(token_b)
    return a is not None and b is not None and a == b


def resolve_column_index(table: dict, column: str) -> int | None:
    headers = table.get("headers") or []
    target = _normalize_for_match(column)
    for i, h in enumerate(headers):
        if _normalize_for_match(h) == target:
            return i
    return None


def validate_location(paper: dict, location: dict) -> bool:
    if location["type"] == "table":
        tables = paper.get("tables", [])
        idx = location["table_index"]
        if not (0 <= idx < len(tables)):
            return False
        table = tables[idx]
        column = location.get("column")
        if column is not None and resolve_column_index(table, column) is None:
            return False
        row_index = location.get("row_index")
        rows = table.get("rows", [])
        if row_index is None or row_index == "all":
            return True
        if isinstance(row_index, int):
            return 0 <= row_index < len(rows)
        return False

    if location["type"] == "section":
        for s in paper.get("sections", []):
            if s.get("heading", "") == location["heading"]:
                paras = s.get("paragraphs", [])
                return 0 <= location["paragraph_index"] < len(paras)
        return False

    return False


def read_location_value(paper: dict, location: dict) -> str | None:
    """Deterministic value read — this is the function that makes
    table-cell-sourced extraction require zero LLM calls: given a resolved
    location, return the actual cell/paragraph text directly, never asking
    a model to retype something already indexable."""
    if location["type"] == "table":
        idx = location["table_index"]
        table = paper["tables"][idx]
        row_index = location.get("row_index")
        column = location.get("column")
        rows = table.get("rows", [])
        if isinstance(row_index, int) and column:
            col_idx = resolve_column_index(table, column)
            if col_idx is not None and 0 <= row_index < len(rows) and col_idx < len(rows[row_index]):
                return str(rows[row_index][col_idx]).strip()
            return None
        if isinstance(row_index, int) and 0 <= row_index < len(rows):
            return " ".join(str(c) for c in rows[row_index])
        return None

    if location["type"] == "section":
        for s in paper.get("sections", []):
            if s.get("heading", "") == location["heading"]:
                paras = s.get("paragraphs", [])
                p_idx = location["paragraph_index"]
                if 0 <= p_idx < len(paras):
                    return paras[p_idx]
        return None

    return None


def _location_source_text(paper: dict, location: dict) -> str:
    """Broader haystack (caption+headers+all relevant rows, or the
    paragraph) for substring-based quote verification, mirroring Stage 2's
    approach — used only as a fallback when numeric_equivalent() cannot be
    applied (e.g. non-numeric evidence)."""
    if location["type"] == "table":
        idx = location["table_index"]
        table = paper["tables"][idx]
        parts = [table.get("caption", ""), " ".join(table.get("headers") or [])]
        row_index = location.get("row_index")
        rows = table.get("rows", [])
        if row_index is None or row_index == "all":
            for row in rows:
                parts.append(" ".join(str(c) for c in row))
        elif isinstance(row_index, int) and 0 <= row_index < len(rows):
            parts.append(" ".join(str(c) for c in rows[row_index]))
        return " ".join(parts)

    if location["type"] == "section":
        val = read_location_value(paper, location)
        return val or ""

    return ""


def reverify_property_item(paper: dict, location: dict, quote: str) -> dict:
    """Stage 3's own re-derivation — never trusts Stage 2's location_verified
    boolean directly. Recovers formatting-only Stage 2 false negatives via
    numeric_equivalent(); genuinely fabricated citations still fail."""
    resolved = validate_location(paper, location)
    actual_value = read_location_value(paper, location) if resolved else None
    verified = False
    if resolved and actual_value is not None:
        if numeric_equivalent(quote, actual_value):
            verified = True
        else:
            verified = _normalize_for_match(quote) in _normalize_for_match(_location_source_text(paper, location))
    return {
        "stage3_location_resolved": resolved,
        "stage3_actual_value": actual_value,
        "stage3_verified": verified,
    }


# ══════════════════════════════════════════════════════════════════════════
# Deterministic source_context construction (table-sourced evidence only —
# prose-sourced identifiers require the not-yet-implemented LLM step and are
# deliberately left None here, never guessed)
# ══════════════════════════════════════════════════════════════════════════

IDENTIFIER_COLUMN_KEYWORDS = {
    "material": {"grade", "designation", "alloy", "material", "steel type", "steel", "code"},
    "specimen": {"specimen", "sample", "label"},
    "processing_condition": {"condition", "treatment", "state", "route", "process"},
}
RECOGNIZED_DEFAULT_TEST_VALUES = {"rt", "room temperature", "room-temperature", "ambient"}
TEST_CONDITION_COLUMN_KEYWORDS = {"temperature", "temp", "strain rate", "strain-rate", "orientation", "direction"}


def _find_identifier_value(table: dict, row_index: int, category: str) -> str | None:
    headers = table.get("headers") or []
    rows = table.get("rows", [])
    if not (isinstance(row_index, int) and 0 <= row_index < len(rows)):
        return None
    row = rows[row_index]
    kws = IDENTIFIER_COLUMN_KEYWORDS[category]
    for col_idx, h in enumerate(headers):
        if any(kw in _normalize_for_match(h) for kw in kws) and col_idx < len(row):
            val = str(row[col_idx]).strip()
            if val:
                return val
    return None


def _find_test_condition_values(table: dict, row_index: int) -> dict | None:
    headers = table.get("headers") or []
    rows = table.get("rows", [])
    if not (isinstance(row_index, int) and 0 <= row_index < len(rows)):
        return None
    row = rows[row_index]
    values = {}
    for col_idx, h in enumerate(headers):
        if any(kw in _normalize_for_match(h) for kw in TEST_CONDITION_COLUMN_KEYWORDS) and col_idx < len(row):
            val = str(row[col_idx]).strip()
            if val:
                values[h.strip()] = val
    return values or None


def build_source_context(paper: dict, location: dict, *, material_grade_label: str | None,
                          processing_state_type: str, group_has_test_condition_evidence: bool) -> dict:
    """Best-effort DETERMINISTIC source_context. material_identifier and
    processing_identifier are populated only from (a) an explicit
    identifier-labeled column in the same row, or (b) the material's own
    grade_label appearing verbatim in the table's caption — both are
    positive, checkable evidence, not an assumption. Absence of either
    leaves the field None; it is never inferred from "there's only one
    material in this paper" at this layer (that fallback exists ONLY in
    resolve_material_identity, as the explicitly weaker
    assumed_single_material tier, which does not pass the gate alone)."""
    is_baseline = processing_state_type == "baseline_or_as_received_state"

    if location["type"] != "table":
        # Prose location: no deterministic identifier extraction available.
        tc_status = "default_implicit" if not group_has_test_condition_evidence else "unresolved"
        return {
            "material_identifier": None, "processing_identifier": None,
            "specimen_identifier": None, "specimen_identifier_unique_state": False,
            "is_baseline_or_as_received": is_baseline,
            "test_condition": {"status": tc_status, "value": None},
            "row_key": None, "table_caption": None,
            "section_context": location.get("heading"),
        }

    idx = location["table_index"]
    table = paper["tables"][idx]
    row_index = location.get("row_index")
    caption = table.get("caption", "")

    material_identifier = _find_identifier_value(table, row_index, "material") if isinstance(row_index, int) else None
    if material_identifier is None and material_grade_label and len(material_grade_label.strip()) >= 3:
        if _normalize_for_match(material_grade_label) in _normalize_for_match(caption):
            material_identifier = material_grade_label
    # PATCH (row-label material identifier fix): if neither the
    # header-keyword nor the caption-substring check resolved an
    # identifier, fall back to the row's own label (first cell) — a real
    # corpus gap (Acta paper Table 3: empty header on the label column,
    # caption never names the specific grade, yet each row's own first
    # cell literally says "DP1"/"DP2"). EXACT normalized match only,
    # never substring — "DP1 / DP2" must never confirm "DP1" merely
    # because it contains it. A blank, multi-material, or non-matching
    # row label leaves material_identifier None, exactly as before this
    # patch — this is an ADDITIONAL deterministic source, never a
    # relaxation of the existing two checks or of resolve_material_
    # identity's own confirmation gate.
    if (material_identifier is None and material_grade_label
            and isinstance(row_index, int) and 0 <= row_index < len(table.get("rows", []))):
        row = table["rows"][row_index]
        if row:
            row_label_cell = str(row[0]).strip()
            if row_label_cell and _normalize_for_match(row_label_cell) == _normalize_for_match(material_grade_label):
                material_identifier = material_grade_label

    processing_identifier = _find_identifier_value(table, row_index, "processing_condition") if isinstance(row_index, int) else None
    specimen_identifier = _find_identifier_value(table, row_index, "specimen") if isinstance(row_index, int) else None

    tc_values = _find_test_condition_values(table, row_index) if isinstance(row_index, int) else None
    if tc_values:
        test_condition = {"status": "confirmed_explicit", "value": tc_values}
    elif not group_has_test_condition_evidence:
        test_condition = {"status": "default_implicit", "value": None}
    else:
        test_condition = {"status": "unresolved", "value": None}

    return {
        "material_identifier": material_identifier,
        "processing_identifier": processing_identifier,
        "specimen_identifier": specimen_identifier,
        "specimen_identifier_unique_state": False,  # deterministic layer cannot establish
                                                      # uniqueness of a specimen label on its
                                                      # own; left False (safe default) pending
                                                      # the LLM step
        "is_baseline_or_as_received": is_baseline,
        "test_condition": test_condition,
        "row_key": (idx, row_index) if isinstance(row_index, int) else None,
        "table_caption": caption,
        "section_context": None,
    }


# ══════════════════════════════════════════════════════════════════════════
# B/C/D/E. THE OBSERVATION-CONSTRUCTION GATE
# ══════════════════════════════════════════════════════════════════════════

def same_row(ctx_a: dict, ctx_b: dict) -> bool:
    return ctx_a.get("row_key") is not None and ctx_a.get("row_key") == ctx_b.get("row_key")


def material_compatible(ctx_a: dict, ctx_b: dict) -> bool:
    ma, mb = ctx_a.get("material_identifier"), ctx_b.get("material_identifier")
    if ma and mb:
        return _normalize_for_match(ma) == _normalize_for_match(mb)
    return False  # missing on either side = insufficient evidence, not a pass


def processing_compatible(ctx_a: dict, ctx_b: dict) -> bool:
    pa, pb = ctx_a.get("processing_identifier"), ctx_b.get("processing_identifier")
    if pa and pb:
        return _normalize_for_match(pa) == _normalize_for_match(pb)
    sa, sb = ctx_a.get("specimen_identifier"), ctx_b.get("specimen_identifier")
    if (sa and sb and ctx_a.get("specimen_identifier_unique_state") and ctx_b.get("specimen_identifier_unique_state")):
        return _normalize_for_match(sa) == _normalize_for_match(sb)
    # PATCH (baseline-merge fix, post-baseline-run): deliberately NO
    # is_baseline_or_as_received branch here. baseline_or_as_received_state
    # is a GROUP-LEVEL PRIOR (build_source_context sets is_baseline_or_as_
    # received for every item in such a group, regardless of location) —
    # never, by itself, positive evidence that two DIFFERENT cited locations
    # share the same specimen/test history. This is what let the confirmed
    # heliyon cross-table merge (standard tensile UTS + RPS pre-strain YS)
    # through the gate. Same-row items never reach this function at all
    # (positively_compatible's same_row() bypass covers them); a standalone
    # single-location baseline item is still labeled via
    # resolve_processing_identity()'s own, separately-corrected logic below.
    return False


def _tc_value_matches_default(value) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        value = " ".join(str(v) for v in value.values())
    return _normalize_for_match(str(value)) in RECOGNIZED_DEFAULT_TEST_VALUES


def test_condition_compatible(ctx_a: dict, ctx_b: dict) -> bool:
    ta, tb = ctx_a["test_condition"], ctx_b["test_condition"]
    if ta["status"] == "unresolved" or tb["status"] == "unresolved":
        return False  # any unresolved side => do not merge, no exceptions
    if ta["status"] == "confirmed_explicit" and tb["status"] == "confirmed_explicit":
        return _normalize_for_match(str(ta["value"])) == _normalize_for_match(str(tb["value"]))
    if ta["status"] == "default_implicit" and tb["status"] == "default_implicit":
        return True
    if ta["status"] == "default_implicit" and tb["status"] == "confirmed_explicit":
        return _tc_value_matches_default(tb["value"])
    if tb["status"] == "default_implicit" and ta["status"] == "confirmed_explicit":
        return _tc_value_matches_default(ta["value"])
    return False


def positively_compatible(ctx_a: dict, ctx_b: dict) -> bool:
    """The reversed-burden-of-proof rule: merging requires POSITIVE evidence
    on all three axes. Same-row is the sole auto-compatible case, because the
    row itself IS the shared identifier."""
    if same_row(ctx_a, ctx_b):
        return True
    return (material_compatible(ctx_a, ctx_b)
            and processing_compatible(ctx_a, ctx_b)
            and test_condition_compatible(ctx_a, ctx_b))


def cluster_property_evidence(items: list[dict]) -> list[list[dict]]:
    """Each item must be positively_compatible with EVERY existing member of
    a cluster to join it (not just the first) — conservative by construction,
    consistent with 'burden of proof on merging.'"""
    clusters: list[list[dict]] = []
    for item in items:
        placed = False
        for cluster in clusters:
            if all(positively_compatible(item["source_context"], other["source_context"]) for other in cluster):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters


def resolve_material_identity(cluster_items: list[dict], material_grade_label: str | None, n_materials_in_paper: int) -> str:
    identifiers = {_normalize_for_match(ci["source_context"]["material_identifier"])
                   for ci in cluster_items if ci["source_context"].get("material_identifier")}
    if len(identifiers) == 1:
        return "confirmed"
    if len(identifiers) > 1:
        return "unresolved"  # internal contradiction; defensive, should not occur post-clustering
    if material_grade_label and n_materials_in_paper == 1:
        return "assumed_single_material"  # deliberately does NOT pass gate F alone
    return "unresolved"


def _context_signature(source_context: dict):
    """A general, content-agnostic proxy for 'which specimen/processing
    context does this location's evidence come from' — the table caption
    for table-sourced evidence, or the section heading for prose-sourced
    evidence (build_source_context sets exactly one of these, never both).
    Used only to detect whether a group's own cited evidence spans more
    than one such context; never to interpret what the caption/heading
    actually says — no keyword matching of any kind."""
    if source_context.get("table_caption"):
        return ("table", _normalize_for_match(source_context["table_caption"]))
    if source_context.get("section_context"):
        return ("section", _normalize_for_match(source_context["section_context"]))
    return None


def _context_signatures(items: list[dict]) -> set:
    return {sig for it in items if (sig := _context_signature(it["source_context"])) is not None}


def resolve_processing_identity(cluster_items: list[dict], processing_state_type: str, group_items: list[dict]) -> str:
    if processing_state_type == "baseline_or_as_received_state" and all(
            ci["source_context"].get("is_baseline_or_as_received") for ci in cluster_items):
        # PATCH (singleton-baseline fix, post-baseline-run): the group-level
        # baseline_or_as_received_state label is a PRIOR, not location-specific
        # proof — it must not by itself confirm baseline for a cluster whose
        # own evidence doesn't positively, uniquely support it. A multi-item
        # cluster (e.g. same-row) confirms baseline when its OWN members
        # internally agree on exactly one context (their shared row already
        # required positive agreement to merge in the first place). A
        # singleton has nothing internal to corroborate it, so it confirms
        # baseline only when the WHOLE group's cited evidence resolves to
        # exactly one context — i.e. nothing else the group cites points at
        # a different table/section, AND there is at least one real context
        # signature to go on (an empty signature set — no captions/headings
        # recorded at all — is "no evidence", not "homogeneous", and must
        # not promote either). A group whose own evidence spans more than
        # one context (like heliyon's standard-tensile table vs. its
        # RPS-pre-strain table) can never let an individual singleton
        # inherit "confirmed_baseline" purely from the group prior.
        if len(cluster_items) > 1:
            if len(_context_signatures(cluster_items)) == 1:
                return "confirmed_baseline"
        else:
            if len(_context_signatures(group_items)) == 1:
                return "confirmed_baseline"
        # falls through to the identifier-based checks below — correctly
        # resolves to "unresolved" absent independent positive evidence

    identifiers = {_normalize_for_match(ci["source_context"]["processing_identifier"])
                   for ci in cluster_items if ci["source_context"].get("processing_identifier")}
    if len(identifiers) == 1:
        return "confirmed_explicit"

    specimens = {_normalize_for_match(ci["source_context"]["specimen_identifier"])
                 for ci in cluster_items
                 if ci["source_context"].get("specimen_identifier") and ci["source_context"].get("specimen_identifier_unique_state")}
    if len(specimens) == 1:
        return "confirmed_explicit"

    # Same-row confirmation applies ONLY when >1 item was actually merged via
    # the same-row bypass — a lone singleton item trivially "shares a row
    # with itself" and must not be granted processing confirmation by that
    # alone; it still needs an explicit identifier or baseline claim.
    row_keys = {ci["source_context"].get("row_key") for ci in cluster_items}
    if len(cluster_items) > 1 and len(row_keys) == 1 and None not in row_keys:
        return "confirmed_baseline" if processing_state_type == "baseline_or_as_received_state" else "confirmed_explicit"

    return "unresolved"


def _property_completeness(properties: dict) -> dict:
    n = len(properties)
    tier = {1: "single_property", 2: "two_properties", 3: "three_properties"}.get(n, f"{n}_properties")
    return {"properties_present": sorted(properties.keys()), "completeness_tier": tier}


# ══════════════════════════════════════════════════════════════════════════
# PATCH (v0.4, additive-only): property-unit provenance. Adds source_column /
# reported_unit / unit_verification_status to every property record, both in
# valid_observation and orphan_property_evidence. Purely additive: does not
# touch numeric value resolution, source-location verification, clustering,
# material/processing/test-condition compatibility, or the emission decision
# — every existing field, on every existing record, is unchanged.
#
# Table evidence: source_column is the raw header string Stage 2 already
# carried through in location["column"] (see _build_table_candidates in
# evidence_prep_v2.py) — read here, never asked of an LLM, never inferred
# from anything else. The unit is then regex-detected from that exact
# string, deterministically.
#
# Prose evidence: the LLM's own reported_unit claim (already present in the
# existing prompt schema and already carried through
# verify_and_filter_llm_output's verified_prose_properties — no prompt text
# changes needed for this) is NEVER trusted by itself. It is accepted only
# if the same unit token is independently, deterministically found inside
# the already span-verified source_span text. An unconfirmed claim never
# reaches reported_unit — it is visible only through the
# "reported_by_model_unverified" status, distinct from genuinely having no
# unit information at all ("not_available").
# ══════════════════════════════════════════════════════════════════════════

_UNIT_TOKEN_PATTERNS = [
    ("GPa", re.compile(r"\bGPa\b", re.IGNORECASE)),
    ("MPa", re.compile(r"\bMPa\b", re.IGNORECASE)),
    ("Pa", re.compile(r"\bPa\b", re.IGNORECASE)),
    ("%", re.compile(r"%|\bpercent\b", re.IGNORECASE)),
]


def _detect_unit_token(text: str | None) -> str | None:
    """Deterministic, case-insensitive, word-boundary unit detection only —
    never a numeric conversion. Word-boundary matching on "Pa" alone
    already excludes the "Pa" inside "MPa"/"GPa" (no boundary exists
    between the preceding "M"/"G" and "P" in those strings), so checking
    order does not affect correctness, only readability. Returns the first
    matching normalized token, or None if none of GPa/MPa/Pa/%/percent is
    present."""
    if not text:
        return None
    for normalized, pattern in _UNIT_TOKEN_PATTERNS:
        if pattern.search(text):
            return normalized
    return None


def resolve_property_unit(source_column: str | None = None, prose_reported_unit: str | None = None,
                           prose_source_span: str | None = None) -> dict:
    """Table branch (source_column given, non-empty): unit is regex-detected
    directly from the raw header string, zero LLM involvement.

    Prose branch (source_column absent): the model's reported_unit claim is
    only accepted if the same unit token is independently found inside the
    already-verified prose_source_span. Never both zero and non-empty at
    once for a single call — process_evidence_group calls this once per
    branch, table XOR prose, per property item."""
    if source_column:
        detected = _detect_unit_token(source_column)
        if detected is not None:
            return {"source_column": source_column, "reported_unit": detected,
                     "unit_verification_status": "verified_from_table_header"}
        return {"source_column": source_column, "reported_unit": None, "unit_verification_status": "not_available"}

    claimed = _detect_unit_token(prose_reported_unit)
    if claimed is None:
        return {"source_column": None, "reported_unit": None, "unit_verification_status": "not_available"}
    if _detect_unit_token(prose_source_span) == claimed:
        return {"source_column": None, "reported_unit": claimed, "unit_verification_status": "verified_from_local_text"}
    return {"source_column": None, "reported_unit": None, "unit_verification_status": "reported_by_model_unverified"}


_EMPTY_UNIT_INFO = {"source_column": None, "reported_unit": None, "unit_verification_status": "not_available"}


def _merge_replicate_unit_info(prop_items: list[dict]) -> dict:
    """Smallest safe deterministic rule for replicate unit provenance: if
    every replicate item independently agrees on the same
    (reported_unit, unit_verification_status) pair, that pair is retained
    (with source_column collapsed too if it's also unanimous, else None,
    since attributing a single "the" header to a merged replicate slot when
    the headers actually differed would overstate what's known). If the
    items disagree on unit/status at all, no unit is silently chosen —
    reported_unit=None with unit_verification_status="not_available" (the
    existing enum value for "no confirmed unit information available",
    reused here rather than inventing a fifth status just for this case)."""
    unit_infos = [it.get("unit_info") or _EMPTY_UNIT_INFO for it in prop_items]
    agreement = {(u["reported_unit"], u["unit_verification_status"]) for u in unit_infos}
    if len(agreement) != 1:
        return dict(_EMPTY_UNIT_INFO)
    reported_unit, status = next(iter(agreement))
    source_columns = {u.get("source_column") for u in unit_infos}
    source_column = next(iter(source_columns)) if len(source_columns) == 1 else None
    return {"source_column": source_column, "reported_unit": reported_unit, "unit_verification_status": status}


def _scope_group_tag(item: dict) -> tuple | None:
    """None for a compatible/unspecified item (the default, unaffected
    grouping path); a (property_scope, loading_mode) tuple ONLY for a
    positively scope-incompatible item -- this is what stops a bulk
    tensile YS and a micropillar/compressive YS in the same cluster from
    ever being folded together as replicates of each other, while never
    splitting the overwhelmingly common 'unspecified' case into spurious
    extra groups."""
    scope, mode = item.get("property_scope"), item.get("loading_mode")
    return (scope, mode) if _is_scope_incompatible(scope, mode) else None


def build_valid_observation(cluster: list[dict], material_identity: str, processing_identity: str) -> dict:
    # Grouping key includes elongation_type (None for UTS/YS, unaffected)
    # so a uniform-elongation item and a total-elongation item in the same
    # cluster are NEVER folded together as replicates of one another --
    # only items that agree on BOTH the canonical property AND (for
    # elongation) the sub-type are considered replicates of each other.
    # The third element does the analogous job for the orthogonal
    # scope/loading-mode dimension (UTS/YS only, None for elongation) --
    # see _scope_group_tag.
    by_property: dict[tuple, list[dict]] = {}
    for item in cluster:
        key = (item["property"], item.get("elongation_type"), _scope_group_tag(item))
        by_property.setdefault(key, []).append(item)

    properties = {}
    for (prop, elongation_type, _scope_tag), prop_items in by_property.items():
        property_scope = prop_items[0].get("property_scope")
        loading_mode = prop_items[0].get("loading_mode")
        prop_key = _scope_qualified_property_key(
            _elongation_property_key(prop, elongation_type), property_scope, loading_mode)
        extra = {"elongation_type": elongation_type} if prop == "elongation" else {}
        if prop in ("UTS", "YS"):
            extra = {**extra, "property_scope": property_scope, "loading_mode": loading_mode}
        if len(prop_items) == 1:
            it = prop_items[0]
            properties[prop_key] = {
                "canonical_property": prop,
                "value": it.get("value"),
                "source_context": it["source_context"],
                "replicate_info": {"is_aggregate": False, "n_replicates_reported": None, "raw_replicate_values": None},
                **(it.get("unit_info") or _EMPTY_UNIT_INFO),
                **extra,
            }
        else:
            # Replicates: never averaged, never split into fake conditions —
            # preserved as raw_replicate_values on this one property slot.
            # (All items sharing this key already agree on elongation_type,
            # by construction of the grouping key above.)
            properties[prop_key] = {
                "canonical_property": prop,
                "value": None,
                "source_context": prop_items[0]["source_context"],
                "replicate_info": {
                    "is_aggregate": False,
                    "n_replicates_reported": len(prop_items),
                    "raw_replicate_values": [it.get("value") for it in prop_items],
                },
                **_merge_replicate_unit_info(prop_items),
                **extra,
            }

    tc_statuses = {it["source_context"]["test_condition"]["status"] for it in cluster}
    test_conditions_status = tc_statuses.pop() if len(tc_statuses) == 1 else "mixed_within_observation"

    return {
        "observation_status": "valid_observation",
        "linkage": {
            "material_identity": material_identity,
            "processing_identity": processing_identity,
            "test_conditions_status": test_conditions_status,
        },
        "property_completeness": _property_completeness(properties),
        "properties": properties,
    }


def build_orphan_evidence(cluster: list[dict], material_identity: str, processing_identity: str) -> dict:
    reasons = []
    if material_identity != "confirmed":
        reasons.append(f"material_identity={material_identity}")
    if processing_identity not in ("confirmed_explicit", "confirmed_baseline"):
        reasons.append(f"processing_identity={processing_identity}")
    return {
        "observation_status": "orphan_property_evidence",
        "linkage": {"material_identity": material_identity, "processing_identity": processing_identity},
        "reason": "; ".join(reasons),
        "properties": [{"property": it["property"], "value": it.get("value"), "source_context": it["source_context"],
                         "elongation_type": it.get("elongation_type"),
                         "property_scope": it.get("property_scope"), "loading_mode": it.get("loading_mode"),
                         **(it.get("unit_info") or _EMPTY_UNIT_INFO)} for it in cluster],
    }


def construct_candidates(items: list[dict], *, processing_state_type: str,
                          material_grade_label: str | None = None, n_materials_in_paper: int = 1) -> list[dict]:
    """F. Emission decision. `items`: [{"property": "UTS"|"YS"|"elongation",
    "value": <already provenance-verified value>, "source_context": {...}}].
    Provenance re-verification (gate A) happens BEFORE this call — items
    passed in here are assumed to have already survived reverify_property_item.
    Returns a list of valid_observation / orphan_property_evidence dicts."""
    clusters = cluster_property_evidence(items)
    results = []
    for cluster in clusters:
        material_identity = resolve_material_identity(cluster, material_grade_label, n_materials_in_paper)
        processing_identity = resolve_processing_identity(cluster, processing_state_type, items)
        if material_identity == "confirmed" and processing_identity in ("confirmed_explicit", "confirmed_baseline"):
            results.append(build_valid_observation(cluster, material_identity, processing_identity))
        else:
            results.append(build_orphan_evidence(cluster, material_identity, processing_identity))
    return results


# ══════════════════════════════════════════════════════════════════════════
# Composition extraction — deterministic, table-sourced evidence only.
# Prose-sourced composition evidence is left pending_llm_extraction: this
# module does not attempt to parse composition statements out of free text.
# ══════════════════════════════════════════════════════════════════════════

ELEMENT_SYMBOLS = {
    "C", "MN", "SI", "AL", "CR", "NI", "MO", "P", "S", "CU", "V", "NB", "TI",
    "N", "FE", "W", "CO", "B", "ZR", "SN", "CA", "O", "H", "AS",
}


def _build_element_record(element: str, raw_value: str, provenance: dict) -> dict:
    qualifier, value, range_low, range_high = "exact", None, None, None
    low = raw_value.lower().strip()
    if low in ("bal", "balance", "bal."):
        qualifier = "balance"
    elif low.startswith("<"):
        qualifier, value = "less_than", parse_numeric_token(raw_value[1:])
    elif low.startswith(">"):
        qualifier, value = "greater_than", parse_numeric_token(raw_value[1:])
    elif "-" in raw_value and parse_numeric_token(raw_value) is None:
        parts = raw_value.split("-")
        if len(parts) == 2:
            lo, hi = parse_numeric_token(parts[0]), parse_numeric_token(parts[1])
            if lo is not None and hi is not None:
                qualifier, range_low, range_high = "range", lo, hi
    else:
        value = parse_numeric_token(raw_value)
        if value is None:
            qualifier = "not_reported"
    return {
        "element": element, "reported_value": raw_value, "reported_unit": "wt.%",
        "value": value, "qualifier": qualifier, "range_low": range_low, "range_high": range_high,
        "provenance": provenance,
    }


def extract_composition_from_table(paper: dict, location: dict) -> list[dict] | None:
    """Returns element records if this composition_evidence location is a
    table with element-symbol columns and an unambiguous single data row;
    None otherwise (caller marks the material's composition as
    pending_llm_extraction rather than guessing)."""
    if location["type"] != "table":
        return None
    idx = location["table_index"]
    table = paper["tables"][idx]
    headers = table.get("headers") or []
    rows = table.get("rows", [])
    row_index = location.get("row_index")
    if row_index in (None, "all"):
        if len(rows) != 1:
            return None
        row_index = 0
    if not (isinstance(row_index, int) and 0 <= row_index < len(rows)):
        return None

    row = rows[row_index]
    elements = []
    for col_idx, h in enumerate(headers):
        symbol = h.strip().upper()
        if symbol in ELEMENT_SYMBOLS and col_idx < len(row):
            raw = str(row[col_idx]).strip()
            if not raw:
                continue
            elements.append(_build_element_record(
                symbol.capitalize(), raw,
                {"location": {**location, "row_index": row_index, "column": h}, "verified": True},
            ))
    return elements or None


# ══════════════════════════════════════════════════════════════════════════
# Local evidence-bundle construction (bounded — one Stage 2 evidence_group,
# never the whole paper) and location-specific span verification
# ══════════════════════════════════════════════════════════════════════════

def _local_paragraph_indices(paper: dict, heading: str, center_idx: int) -> list[int]:
    for s in paper.get("sections", []):
        if s.get("heading", "") == heading:
            n = len(s.get("paragraphs", []))
            lo = max(0, center_idx - LOCAL_PARAGRAPH_WINDOW)
            hi = min(n - 1, center_idx + LOCAL_PARAGRAPH_WINDOW)
            return list(range(lo, hi + 1))
    return []


def _render_bundle_tables(paper: dict, table_indices: set[int]) -> str:
    out = ""
    for i in sorted(table_indices):
        t = paper["tables"][i]
        out += f"\nTable {i} (index {i}): {t.get('caption', '')}\n"
        if t.get("headers"):
            out += "  Headers: " + " | ".join(t["headers"]) + "\n"
        for r_idx, row in enumerate(t.get("rows", [])):
            out += f"  [row {r_idx}] " + " ".join(str(c) for c in row) + "\n"
    return out


def _render_bundle_sections(paper: dict, section_windows: dict[str, set[int]]) -> str:
    out = ""
    for heading, idxs in section_windows.items():
        for s in paper.get("sections", []):
            if s.get("heading", "") == heading:
                out += f"\n[Section] {heading}\n"
                paras = s.get("paragraphs", [])
                for p_idx in sorted(idxs):
                    if 0 <= p_idx < len(paras):
                        out += f"  Paragraph {p_idx}: {paras[p_idx]}\n"
    return out


def build_local_bundle(paper: dict, material: dict, group: dict) -> dict:
    """Bounded to exactly what this evidence_group already cites: whole
    table(s) it references (no other tables), and a +/-LOCAL_PARAGRAPH_WINDOW
    window around each cited paragraph within the SAME section only (never
    crossing a section heading, never reaching a section/table not cited by
    this group). allowed_locations is the hard filter bounded local discovery
    and span verification are checked against.

    PATCH (crash fix, post-baseline-run): every cited location is validated
    via validate_location() BEFORE being used to index into paper["tables"]/
    paper["sections"] — a Stage 2 citation can carry an out-of-range
    table_index (confirmed on dt-2022-10-010, already location_verified=False
    in Stage 2's own frozen output) and previously crashed this function with
    an unhandled IndexError. Invalid locations are now skipped and returned
    in invalid_locations for the caller to record as rejected candidates."""
    all_items = (list(group.get("processing_evidence", []))
                 + list(group.get("test_condition_evidence", []))
                 + list(group.get("property_evidence", [])))

    table_indices: set[int] = set()
    section_windows: dict[str, set[int]] = {}
    invalid_locations: list[dict] = []
    for it in all_items:
        loc = it["location"]
        if not validate_location(paper, loc):
            invalid_locations.append(loc)
            continue
        if loc["type"] == "table":
            table_indices.add(loc["table_index"])
        elif loc["type"] == "section":
            idxs = _local_paragraph_indices(paper, loc["heading"], loc["paragraph_index"])
            section_windows.setdefault(loc["heading"], set()).update(idxs)

    allowed_locations = {("table", ti) for ti in table_indices}
    for heading, idxs in section_windows.items():
        allowed_locations |= {("section", heading, p) for p in idxs}

    return {
        "material_id": material["material_id"],
        "group_id": group["group_id"],
        "grade_label_prior": material.get("grade_label", ""),
        "processing_state_type_prior": group.get("processing_state_type", "not_established"),
        "allowed_locations": allowed_locations,
        "tables_text": _render_bundle_tables(paper, table_indices),
        "sections_text": _render_bundle_sections(paper, section_windows),
        "invalid_locations": invalid_locations,
    }


def bundle_has_usable_content(bundle: dict) -> bool:
    """PATCH (post-baseline-run): a bundle with no allowed locations at all
    (every cited location was invalid, or a cited section resolved to an
    empty local window — jmrt-2020-06-063's exact case: a 0-paragraph
    section) has nothing real to show the LLM. Calling it anyway invites
    hallucination (confirmed on jmrt: the model returned a plausible-looking
    but unverifiable span for content it was never actually shown) and
    wastes a call. Checked before run_llm_extraction_pass() is ever invoked."""
    return bool(bundle["allowed_locations"])


def _location_allowed(bundle: dict, loc: dict) -> bool:
    if loc["type"] == "table":
        return ("table", loc["table_index"]) in bundle["allowed_locations"]
    if loc["type"] == "section":
        return ("section", loc["heading"], loc["paragraph_index"]) in bundle["allowed_locations"]
    return False


def verify_llm_source_span(paper: dict, location: dict, source_span: str) -> bool:
    """LOCATION-SPECIFIC verification against the original processed-paper
    JSON (the source of truth), never against the whole bundle text. Requires
    BOTH: the location resolves to a real element, AND the normalized span
    occurs inside the text represented by THAT SPECIFIC location only.
    _location_source_text() already scopes to exactly one row (or one
    paragraph) when an int row_index/paragraph_index is given — it does not
    search other rows/paragraphs — so a span claimed at paragraph 5 is
    checked only against paragraph 5's own text, never paragraph 2's, even
    if both are in the same bundle."""
    if not validate_location(paper, location):
        return False
    needle = _normalize_for_match(source_span)
    if not needle:
        return False
    haystack = _normalize_for_match(_location_source_text(paper, location))
    return needle in haystack


# ══════════════════════════════════════════════════════════════════════════
# Structured Stage 3 SYSTEM_PROMPT (first drafted version — not yet run
# against a real paper, not yet reviewed against real output samples the way
# Stage 1/Stage 2's prompts were before being trusted)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are resolving LOCAL context for one evidence group from a steel
materials science paper, for a deterministic extraction pipeline. You are
shown only a small, already-selected excerpt — one table and/or a short
window of nearby paragraphs — never the whole paper.

CRITICAL RULE — DO NOT RETYPE TABLE VALUES: every numeric value in any table
shown to you has already been read by deterministic code. Do not report a
table-cell value anywhere in your response. Only report a numeric property
value if it occurs in PROSE (paragraph) text shown to you, never in a table.

CRITICAL RULE — QUOTE EXACTLY AND LOCATE PRECISELY: every source_span you
report must be copied EXACTLY (verbatim) from the specific paragraph or
table you cite for it, and every location you cite must be part of what was
shown to you. A source_span will be rejected if it does not literally
appear at the exact location you claim it came from — even if the same
phrase appears elsewhere in what you were shown.

CRITICAL RULE — ONLY WHAT'S SHOWN: you may only cite locations that are
part of this bundle. Do not reference or guess at any other table or
paragraph not shown to you here, even if you believe one might exist.

CRITICAL RULE ON UNCERTAINTY: if you cannot find a positive textual link
between a value and a material name or processing condition IN WHAT YOU
WERE SHOWN, report status "not_found_in_local_text". Do not guess, and do
not treat "this paper only seems to discuss one material" as sufficient
grounds for "confirmed_in_local_text" — you must point at actual text.

Return ONLY a valid JSON object with this structure — no extra text:
{
  "material_identifier_resolution": {"identifier": "...", "status": "confirmed_in_local_text"|"not_found_in_local_text", "source_span": "...", "location": {...}} | null,
  "processing_identifier_resolution": {"identifier": "...", "status": "confirmed_in_local_text"|"not_found_in_local_text", "source_span": "...", "location": {...}} | null,
  "stage3_processing_state_type": "explicit_processing_condition" | "baseline_or_as_received_state" | "not_established",
  "processing_route": [
    {"step_index": 1, "process_type": "intercritical_annealing", "reported_process_name": "intercritical annealing",
     "temperature": {"reported_value": "700", "reported_unit": "°C", "source_span": "..."} , "time": null, "location": {...}}
  ],
  "test_condition_identifiers": [
    {"parameter": "test_temperature", "reported_value": "RT", "source_span": "...", "location": {...}}
  ],
  "prose_property_candidates": [
    {"property": "YS", "reported_property_name": "yield strength", "reported_numeric_token": "1465",
     "reported_unit": "MPa", "source_span": "yield strength of 1465 MPa", "location": {...}}
  ],
  "linkage_cues": [
    {"cue_type": "shared_specimen_identifier", "description": "one sentence",
     "involved_locations": [{...}], "source_span": "..."}
  ],
  "additional_local_condition_candidates": [
    {"reason": "one sentence", "process_type": "quenching_and_partitioning", "source_span": "...", "location": {...}}
  ],
  "notes": "one sentence",
  "confidence": "high" | "medium" | "low"
}

Valid location forms:
  table:   {"type":"table","table_index":int,"row_index":int|null,"column":str|null}
  section: {"type":"section","heading":str,"paragraph_index":int}

Valid process_type values: intercritical_annealing | quenching | partitioning
| tempering | austempering | ausforming | austenitizing | annealing |
normalizing | rolling | cooling | reheating | soaking | other_explicit

Valid property values (prose_property_candidates only): UTS | YS | elongation

Leave any list empty ([]) if nothing applies. Leave
material_identifier_resolution or processing_identifier_resolution null if
nothing in the local text resolves them — do not fabricate one.
"""


def build_stage3_prompt(bundle: dict) -> str:
    prior = (f"Stage 2 prior (a prior guess, not given truth): material grade_label="
             f"{bundle['grade_label_prior']!r}, processing_state_type={bundle['processing_state_type_prior']!r}. "
             f"Confirm or contradict this from the local text below; do not assume it is correct.")
    return f"""{SYSTEM_PROMPT}

{prior}

TABLES (full content — do not retype any cell value from these):
{bundle['tables_text'] if bundle['tables_text'] else '(no tables in this bundle)'}

PARAGRAPHS (local window only):
{bundle['sections_text'] if bundle['sections_text'] else '(no section text in this bundle)'}

Respond with ONLY the JSON object now. No other text.
"""


# ══════════════════════════════════════════════════════════════════════════
# Defensive normalization of the raw LLM response (same policy as Stage 1/2:
# never crash on malformed input, always produce a well-typed structure with
# safe defaults)
# ══════════════════════════════════════════════════════════════════════════

VALID_PROCESS_TYPES = {
    "intercritical_annealing", "quenching", "partitioning", "tempering", "austempering",
    "ausforming", "austenitizing", "annealing", "normalizing", "rolling", "cooling",
    "reheating", "soaking", "other_explicit",
}
VALID_STAGE3_PROCESSING_STATE_TYPE = {"explicit_processing_condition", "baseline_or_as_received_state", "not_established"}
VALID_CUE_TYPE = {"shared_specimen_identifier", "shared_processing_identifier", "explicit_cross_reference", "contradiction"}
VALID_RESOLUTION_STATUS = {"confirmed_in_local_text", "not_found_in_local_text"}
VALID_LLM_CONFIDENCE = {"high", "medium", "low"}
VALID_PROSE_PROPERTY = {"UTS", "YS", "elongation"}

# Self-contained copy of evidence_prep_v2's elongation sub-type classifier
# (same pattern already used for call_llm/_post_to_ollama in this file --
# each stage's copy is independent so neither depends on the other's
# internals). Conservative, source-wording-only -- 'total' | 'uniform' |
# 'fracture' | 'unspecified'; never inferred from numeric magnitude or
# table position; a bare "Elongation (%)" is 'unspecified', never silently
# promoted to 'total'.
_ELONGATION_FRACTURE_RE = re.compile(
    r"elongation\s*(?:at|to)\s*(?:break|failure)|elongation\s+after\s+(?:break|fracture)|fracture\s+elongation",
    re.IGNORECASE)
_ELONGATION_TOTAL_RE = re.compile(r"total\s+elong|\bTE\b", re.IGNORECASE)
_ELONGATION_UNIFORM_RE = re.compile(r"uniform\s+elong|\bUE\b|\bA\s?g\b|\bA_g\b", re.IGNORECASE)


def classify_elongation_type(text: str) -> str:
    text = text or ""
    if _ELONGATION_FRACTURE_RE.search(text):
        return "fracture"
    if _ELONGATION_TOTAL_RE.search(text):
        return "total"
    if _ELONGATION_UNIFORM_RE.search(text):
        return "uniform"
    return "unspecified"


def _elongation_property_key(property_name: str, elongation_type: str | None) -> str:
    """The properties-dict key for one item: for UTS/YS this is simply the
    property name unchanged (no behaviour change). For elongation, the key
    is type-qualified (elongation_total/elongation_uniform/
    elongation_fracture/elongation_unspecified) so that a uniform-elongation
    and a total-elongation value from the SAME observation occupy separate
    dict slots instead of colliding on one "elongation" key -- this is what
    stops them from ever being treated as replicates of each other."""
    if property_name != "elongation":
        return property_name
    return f"elongation_{elongation_type or 'unspecified'}"


# ══════════════════════════════════════════════════════════════════════════
# Property scope (measurement level) and loading mode (tensile vs
# compressive) -- self-contained copy of Stage 2's classifiers (matches
# the established per-stage duplication convention already used for
# classify_elongation_type). Source-wording-only, from the column header
# text / reported_property_name + verified source_span alone, never from
# the table caption -- see evidence_prep_v2.classify_property_scope's
# docstring for the real Acta-paper rationale (a table caption mentioning
# both macro and micro scale must not make genuinely bulk data ambiguous).
# ══════════════════════════════════════════════════════════════════════════
_LOADING_COMPRESSIVE_RE = re.compile(r"compressive|compression", re.IGNORECASE)
_LOADING_TENSILE_RE = re.compile(r"\btensile\b|\btension\b", re.IGNORECASE)
_SCOPE_MICRO_RE = re.compile(r"micro[- ]?pillar|micropillar|\bpillars?\b|micro[- ]?scale|nano[- ]?indent", re.IGNORECASE)
_SCOPE_MACRO_RE = re.compile(r"macro[- ]?scale|\bmacro\b|\bbulk\b", re.IGNORECASE)
_PHASE_COMPONENT_RE = re.compile(r"\b(ferrite|martensite|austenite|bainite|pearlite|cementite)\b", re.IGNORECASE)


def classify_loading_mode(text: str) -> str:
    text = text or ""
    is_compressive = bool(_LOADING_COMPRESSIVE_RE.search(text))
    is_tensile = bool(_LOADING_TENSILE_RE.search(text))
    if is_compressive and is_tensile:
        return "unspecified"
    if is_compressive:
        return "compressive"
    if is_tensile:
        return "tensile"
    return "unspecified"


def classify_property_scope(text: str) -> str:
    text = text or ""
    if _PHASE_COMPONENT_RE.search(text):
        return "phase_level"
    has_micro = bool(_SCOPE_MICRO_RE.search(text))
    has_macro = bool(_SCOPE_MACRO_RE.search(text))
    if has_micro and has_macro:
        return "unspecified"
    if has_micro:
        return "microscale_or_local"
    if has_macro:
        return "bulk_or_macroscale"
    return "unspecified"


def _is_scope_incompatible(property_scope: str | None, loading_mode: str | None) -> bool:
    """True only for a POSITIVELY detected incompatibility with the
    bulk/macroscale-tensile modelling targets -- compressive loading, or a
    microscale/local/phase-level measurement. 'unspecified' in either
    dimension is never incompatible by itself (see the module-level design
    note in build_valid_observation): absence of a cue must not be treated
    as a detected problem, only a present, positive cue is."""
    return loading_mode == "compressive" or property_scope in ("microscale_or_local", "phase_level")


def _scope_qualified_property_key(property_name: str, property_scope: str | None, loading_mode: str | None) -> str:
    """Mirrors _elongation_property_key's role for the orthogonal
    scope/loading-mode dimension: UTS/YS stays keyed by its plain name
    (zero behaviour change) UNLESS its scope/mode is positively
    incompatible with the bulk-tensile target, in which case it is
    diverted to a separate, distinctly-named slot (e.g.
    'YS_phase_level_compressive') so it is never silently averaged
    together with (or displaces) genuine bulk-tensile replicates of the
    same nominal property name -- the exact failure mode a micropillar
    compressive yield-strength value being folded into a bulk tensile YS
    would otherwise cause."""
    if not _is_scope_incompatible(property_scope, loading_mode):
        return property_name
    return f"{property_name}_{property_scope or 'unspecified'}_{loading_mode or 'unspecified'}"


def _normalize_location_field(loc) -> dict | None:
    if not isinstance(loc, dict):
        return None
    ltype = loc.get("type")
    if ltype == "table":
        idx = loc.get("table_index")
        if not isinstance(idx, int):
            return None
        return {"type": "table", "table_index": idx, "row_index": loc.get("row_index"), "column": loc.get("column")}
    if ltype == "section":
        heading, p_idx = loc.get("heading"), loc.get("paragraph_index")
        if not isinstance(heading, str) or not isinstance(p_idx, int):
            return None
        return {"type": "section", "heading": heading, "paragraph_index": p_idx}
    return None


def _normalize_resolution(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    loc = _normalize_location_field(raw.get("location"))
    span, identifier, status = raw.get("source_span"), raw.get("identifier"), raw.get("status")
    if loc is None or not isinstance(span, str) or not span.strip() or not isinstance(identifier, str) or not identifier.strip():
        return None
    if status not in VALID_RESOLUTION_STATUS:
        status = "not_found_in_local_text"
    return {"identifier": identifier, "status": status, "source_span": span, "location": loc}


def _normalize_numeric_field(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    span, val = raw.get("source_span"), raw.get("reported_value")
    if not isinstance(span, str) or not span.strip() or val is None:
        return None
    return {"reported_value": val, "reported_unit": raw.get("reported_unit"), "source_span": span}


def _normalize_processing_step(raw, idx: int) -> dict | None:
    if not isinstance(raw, dict):
        return None
    loc = _normalize_location_field(raw.get("location"))
    if loc is None:
        return None
    process_type = raw.get("process_type") if raw.get("process_type") in VALID_PROCESS_TYPES else "other_explicit"
    return {
        "step_index": raw.get("step_index") if isinstance(raw.get("step_index"), int) else idx,
        "process_type": process_type,
        "reported_process_name": raw.get("reported_process_name") if isinstance(raw.get("reported_process_name"), str) else "",
        "temperature": _normalize_numeric_field(raw.get("temperature")),
        "time": _normalize_numeric_field(raw.get("time")),
        "location": loc,
    }


def _normalize_test_condition_identifier(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    loc = _normalize_location_field(raw.get("location"))
    span = raw.get("source_span")
    if loc is None or not isinstance(span, str) or not span.strip():
        return None
    return {
        "parameter": raw.get("parameter") if isinstance(raw.get("parameter"), str) else "unspecified",
        "reported_value": raw.get("reported_value"), "source_span": span, "location": loc,
    }


def _normalize_prose_property_candidate(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    loc = _normalize_location_field(raw.get("location"))
    span, prop, token = raw.get("source_span"), raw.get("property"), raw.get("reported_numeric_token")
    if (loc is None or loc["type"] != "section" or prop not in VALID_PROSE_PROPERTY
            or not isinstance(span, str) or not span.strip() or token is None):
        return None
    return {
        "property": prop,
        "reported_property_name": raw.get("reported_property_name") if isinstance(raw.get("reported_property_name"), str) else prop,
        "reported_numeric_token": token, "reported_unit": raw.get("reported_unit"),
        "source_span": span, "location": loc,
    }


def _normalize_linkage_cue(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    cue_type, span, involved = raw.get("cue_type"), raw.get("source_span"), raw.get("involved_locations")
    if cue_type not in VALID_CUE_TYPE or not isinstance(span, str) or not span.strip() or not isinstance(involved, list):
        return None
    locs = [n for l in involved if (n := _normalize_location_field(l)) is not None]
    if not locs:
        return None
    return {
        "cue_type": cue_type,
        "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
        "involved_locations": locs, "source_span": span,
    }


def _normalize_discovery_candidate(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    loc = _normalize_location_field(raw.get("location"))
    span = raw.get("source_span")
    if loc is None or not isinstance(span, str) or not span.strip():
        return None
    process_type = raw.get("process_type") if raw.get("process_type") in VALID_PROCESS_TYPES else "other_explicit"
    return {
        "reason": raw.get("reason") if isinstance(raw.get("reason"), str) else "",
        "process_type": process_type, "source_span": span, "location": loc,
    }


def normalize_llm_output(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    stage3_pst = raw.get("stage3_processing_state_type")
    if stage3_pst not in VALID_STAGE3_PROCESSING_STATE_TYPE:
        stage3_pst = "not_established"
    confidence = raw.get("confidence") if raw.get("confidence") in VALID_LLM_CONFIDENCE else "low"
    return {
        "material_identifier_resolution": _normalize_resolution(raw.get("material_identifier_resolution")),
        "processing_identifier_resolution": _normalize_resolution(raw.get("processing_identifier_resolution")),
        "stage3_processing_state_type": stage3_pst,
        "processing_route": [n for i, r in enumerate(raw.get("processing_route") or [], 1)
                              if (n := _normalize_processing_step(r, i)) is not None],
        "test_condition_identifiers": [n for r in (raw.get("test_condition_identifiers") or [])
                                        if (n := _normalize_test_condition_identifier(r)) is not None],
        "prose_property_candidates": [n for r in (raw.get("prose_property_candidates") or [])
                                       if (n := _normalize_prose_property_candidate(r)) is not None],
        "linkage_cues": [n for r in (raw.get("linkage_cues") or []) if (n := _normalize_linkage_cue(r)) is not None],
        "additional_local_condition_candidates": [n for r in (raw.get("additional_local_condition_candidates") or [])
                                                    if (n := _normalize_discovery_candidate(r)) is not None],
        "confidence": confidence,
        "notes": raw.get("notes") if isinstance(raw.get("notes"), str) else "",
    }


def parse_llm_response(text) -> dict | None:
    if not isinstance(text, str):
        return None
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ══════════════════════════════════════════════════════════════════════════
# Verification/filtering of normalized LLM output against the original
# paper JSON — every span, every location, every numeric token is
# independently re-checked here; nothing normalize_llm_output produced is
# trusted just because it parsed.
# ══════════════════════════════════════════════════════════════════════════

def verify_and_filter_llm_output(paper: dict, bundle: dict, normalized: dict) -> dict:
    def span_ok(loc, span):
        return _location_allowed(bundle, loc) and verify_llm_source_span(paper, loc, span)

    verified_material_identifier = None
    mr = normalized["material_identifier_resolution"]
    if mr and mr["status"] == "confirmed_in_local_text" and span_ok(mr["location"], mr["source_span"]):
        verified_material_identifier = mr["identifier"]

    verified_processing_identifier = None
    pr = normalized["processing_identifier_resolution"]
    if pr and pr["status"] == "confirmed_in_local_text" and span_ok(pr["location"], pr["source_span"]):
        verified_processing_identifier = pr["identifier"]

    verified_route = []
    for step in normalized["processing_route"]:
        if not _location_allowed(bundle, step["location"]):
            continue
        temp = step.get("temperature")
        if temp:
            if not verify_llm_source_span(paper, step["location"], temp["source_span"]):
                temp = None
            else:
                temp = {**temp, "value": parse_numeric_token(temp["reported_value"])}
        time_ = step.get("time")
        if time_:
            if not verify_llm_source_span(paper, step["location"], time_["source_span"]):
                time_ = None
            else:
                time_ = {**time_, "value": parse_numeric_token(time_["reported_value"])}
        verified_route.append({**step, "temperature": temp, "time": time_})

    verified_test_conditions = [tc for tc in normalized["test_condition_identifiers"]
                                 if span_ok(tc["location"], tc["source_span"])]

    verified_prose_properties, rejected_prose_properties = [], []
    for cand in normalized["prose_property_candidates"]:
        if not span_ok(cand["location"], cand["source_span"]):
            rejected_prose_properties.append({**cand, "reason": "unverified_source_span"})
            continue
        parsed = parse_numeric_token(cand["reported_numeric_token"])
        if parsed is None:
            rejected_prose_properties.append({**cand, "reason": "unsupported_numerical_token"})
            continue
        verified_prose_properties.append({**cand, "value": parsed})

    verified_linkage_cues = [
        cue for cue in normalized["linkage_cues"]
        if all(_location_allowed(bundle, loc) and validate_location(paper, loc) for loc in cue["involved_locations"])
        and any(verify_llm_source_span(paper, loc, cue["source_span"]) for loc in cue["involved_locations"])
    ]

    verified_discoveries, rejected_discoveries = [], []
    for cand in normalized["additional_local_condition_candidates"]:
        if span_ok(cand["location"], cand["source_span"]):
            verified_discoveries.append(cand)
        else:
            rejected_discoveries.append({**cand, "reason": "unverified_source_span_or_out_of_local_window"})

    return {
        "verified_material_identifier": verified_material_identifier,
        "verified_processing_identifier": verified_processing_identifier,
        "stage3_processing_state_type": normalized["stage3_processing_state_type"],
        "verified_processing_route": verified_route,
        "verified_test_conditions": verified_test_conditions,
        "verified_prose_properties": verified_prose_properties,
        "rejected_prose_properties": rejected_prose_properties,
        "verified_linkage_cues": verified_linkage_cues,
        "verified_discoveries": verified_discoveries,
        "rejected_discoveries": rejected_discoveries,
        "confidence": normalized["confidence"],
        "notes": normalized["notes"],
    }


def derive_confirmed_unique_specimens(paper: dict, verified_output: dict) -> set[str]:
    """Deterministic re-derivation, not text parsing of the LLM's free-text
    description: for each verified shared_specimen_identifier cue, read the
    ACTUAL specimen-identifier column value at each involved table location."""
    confirmed = set()
    for cue in verified_output["verified_linkage_cues"]:
        if cue["cue_type"] != "shared_specimen_identifier":
            continue
        for loc in cue["involved_locations"]:
            if loc["type"] == "table":
                val = _find_identifier_value(paper["tables"][loc["table_index"]], loc.get("row_index"), "specimen")
                if val:
                    confirmed.add(_normalize_for_match(val))
    return confirmed


# ══════════════════════════════════════════════════════════════════════════
# Additive merge: fills nulls in a deterministic source_context from
# LLM-verified group-level identifiers. Never overwrites an already-populated
# field; specimen_identifier_unique_state may only move False -> True.
# ══════════════════════════════════════════════════════════════════════════

def merge_llm_context_into_source_context(deterministic_ctx: dict, llm_group_identifiers: dict) -> dict:
    ctx = dict(deterministic_ctx)
    if ctx.get("material_identifier") is None and llm_group_identifiers.get("material_identifier"):
        ctx["material_identifier"] = llm_group_identifiers["material_identifier"]
    if ctx.get("processing_identifier") is None and llm_group_identifiers.get("processing_identifier"):
        ctx["processing_identifier"] = llm_group_identifiers["processing_identifier"]
    if not ctx.get("specimen_identifier_unique_state") and ctx.get("specimen_identifier"):
        confirmed = llm_group_identifiers.get("confirmed_unique_specimen_identifiers") or set()
        if _normalize_for_match(ctx["specimen_identifier"]) in confirmed:
            ctx["specimen_identifier_unique_state"] = True
    return ctx


# ══════════════════════════════════════════════════════════════════════════
# LLM call scaffold — the network call itself is not exercised by the
# offline test suite (see test_stage3_gate.py); process_llm_response_for_group
# below is the pure, directly-testable core that everything except the
# actual HTTP call routes through.
#
# PATCH (timeout-mechanism fix, post-baseline-run): the previous
# ThreadPoolExecutor-as-context-manager implementation did NOT bound caller
# latency — confirmed on jmrt-2020-06-063 (a call ran 697.8s against a
# configured 260s cap) and reproduced in isolation (a 1s watchdog around a
# 3s task still took the full 3s to propagate). The reason:
# ThreadPoolExecutor.__exit__ calls shutdown(wait=True) on ANY exit from the
# `with` block, which blocks until the submitted (already-running,
# uncancellable) task actually finishes, regardless of whether
# future.result(timeout=...) already raised TimeoutError.
#
# Fixed with a plain daemon thread + Thread.join(timeout=...), whose timeout
# semantics are genuine: join() returns control to the caller after AT MOST
# the given duration, independent of whether the thread is still running.
# This is a CALLER-SIDE hard wall-clock deadline only — it bounds how long
# the Stage 3 pipeline waits on this call, and lets processing move on to
# the next group/paper. It does NOT terminate the underlying Ollama
# generation: an abandoned request keeps running server-side and in its own
# background thread, whose (now-unread) result is simply discarded, exactly
# as one would "hang up" on a slow call without being able to stop what's
# happening on the other end. True process-level cancellation (killing the
# Ollama-side generation itself) is out of scope for this round — flagged
# here as a scaling limitation to revisit before any full-corpus execution,
# since abandoned background requests could otherwise compete with later
# calls for Ollama's single active-generation slot.
# ══════════════════════════════════════════════════════════════════════════

class LLMCallError(Exception):
    pass


def _post_to_ollama(prompt: str):
    return requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 32768},
    }, timeout=LLM_CALL_TIMEOUT_S)


def call_llm(prompt: str) -> str:
    """Caller-side hard wall-clock deadline via a daemon thread. See the
    module-level PATCH note above for why this replaced the previous
    ThreadPoolExecutor-based implementation and what it does/doesn't bound."""
    result_box: dict = {}

    def worker():
        try:
            result_box["resp"] = _post_to_ollama(prompt)
        except requests.exceptions.RequestException as e:
            result_box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=LLM_WALLCLOCK_HARD_TIMEOUT_S)

    if t.is_alive():
        # Genuine timeout: return to the caller now. The worker thread is
        # abandoned — never joined again — and, being a daemon thread, will
        # be killed outright at process exit rather than blocking it. If it
        # later completes in the background, nothing reads result_box again.
        raise LLMCallError(f"caller-side hard wall-clock timeout after {LLM_WALLCLOCK_HARD_TIMEOUT_S}s (request abandoned, not cancelled server-side)")

    if "error" in result_box:
        raise LLMCallError(f"request failed: {type(result_box['error']).__name__}: {result_box['error']}")

    resp = result_box["resp"]
    try:
        body = resp.json()
    except ValueError:
        raise LLMCallError(f"non-JSON HTTP response (status {resp.status_code}): {resp.text[:200]!r}")
    if resp.status_code != 200:
        raise LLMCallError(f"HTTP {resp.status_code}: {body}")
    if not isinstance(body, dict) or "response" not in body:
        raise LLMCallError(f"no 'response' field (status {resp.status_code}): {body}")
    text = body["response"]
    if not isinstance(text, str):
        raise LLMCallError(f"'response' field was not a string (got {type(text).__name__}): {text!r}")
    return text


def call_llm_with_retry(prompt: str, max_attempts: int = LLM_MAX_ATTEMPTS) -> str:
    last_error = None
    for _ in range(max_attempts):
        try:
            return call_llm(prompt)
        except LLMCallError as e:
            last_error = e
    raise LLMCallError(f"all {max_attempts} attempts failed") from last_error


def process_llm_response_for_group(paper: dict, bundle: dict, raw_text: str | None, call_error: str | None = None) -> dict:
    """Pure, network-free core: given a raw LLM response (or a call_error
    standing in for one), produce a status + verified_output. This is the
    function the offline test suite exercises directly for the malformed-
    JSON and timeout/failure paths, without ever touching the network."""
    if call_error is not None:
        return {"stage3_llm_status": "timeout", "verified_output": None, "raw_error": call_error}
    if raw_text is None:
        return {"stage3_llm_status": "no_response", "verified_output": None, "raw_error": None}
    raw_json = parse_llm_response(raw_text)
    if raw_json is None:
        return {"stage3_llm_status": "malformed_json", "verified_output": None, "raw_error": raw_text[:300]}
    normalized = normalize_llm_output(raw_json)
    verified = verify_and_filter_llm_output(paper, bundle, normalized)
    return {"stage3_llm_status": "ok", "verified_output": verified, "raw_error": None}


def run_llm_extraction_pass(paper: dict, material: dict, group: dict, bundle: dict) -> dict:
    """The only function in this module that actually calls the LLM.
    PATCH (post-baseline-run): now takes an already-built bundle rather than
    building one internally — process_evidence_group builds it once, uses it
    to check invalid_locations/bundle_has_usable_content(), and only then
    calls this."""
    prompt = build_stage3_prompt(bundle)
    try:
        raw_text = call_llm_with_retry(prompt)
    except LLMCallError as e:
        return process_llm_response_for_group(paper, bundle, None, call_error=str(e))
    return process_llm_response_for_group(paper, bundle, raw_text)


# ══════════════════════════════════════════════════════════════════════════
# Per-evidence-group processing: wires deterministic re-verification +
# (optionally) the LLM local-context layer into the frozen gate. With
# llm_enabled=False (the default, and the only mode used anywhere in this
# session), only deterministic table-sourced evidence is processed — no
# network call is attempted at all.
# ══════════════════════════════════════════════════════════════════════════

def process_evidence_group(paper: dict, material: dict, group: dict, n_materials_in_paper: int, llm_enabled: bool = False) -> dict:
    group_id = group["group_id"]
    material_id = material["material_id"]
    processing_state_type = group.get("processing_state_type", "not_established")
    grade_label = material.get("grade_label", "")
    has_test_condition_evidence = bool(group.get("test_condition_evidence"))

    rejected: list[dict] = []
    failures: list[dict] = []
    items: list[dict] = []

    for pe in group.get("property_evidence", []):
        loc = pe["location"]
        rv = reverify_property_item(paper, loc, pe.get("quote", ""))
        if not rv["stage3_verified"]:
            rejected.append({"property": pe.get("property"), "location": loc,
                              "reason": "provenance_reverification_failed", "stage2_quote": pe.get("quote", "")})
            continue
        value = parse_numeric_token(rv["stage3_actual_value"])
        if value is None:
            rejected.append({"property": pe.get("property"), "location": loc,
                              "reason": "unsupported_numerical_token", "actual_value": rv["stage3_actual_value"]})
            continue
        ctx = build_source_context(paper, loc, material_grade_label=grade_label,
                                    processing_state_type=processing_state_type,
                                    group_has_test_condition_evidence=has_test_condition_evidence)
        # PATCH (v0.4, additive): source_column is the raw header Stage 2
        # already put in loc["column"] -- read here, never inferred, never
        # asked of an LLM. See resolve_property_unit's docstring.
        # PATCH (row-oriented fallback, additive): for a row-oriented
        # candidate, loc["column"] is deliberately the REAL column header
        # (e.g. "AB") -- required unchanged for reverify_property_item/
        # resolve_column_index to keep relocating the cell correctly -- so
        # it carries no unit. Stage 2 supplies the row's own unit-bearing
        # label separately as "unit_source_text" (present only for
        # row-oriented candidates; absent, i.e. None, for every existing
        # column-oriented item, which falls through to the unchanged
        # loc["column"] behavior exactly as before this patch).
        unit_info = resolve_property_unit(source_column=pe.get("unit_source_text") or loc.get("column"))
        # elongation_type is Stage 2's own classification (from the source
        # header text, e.g. "UE [%]"/"Total elongation (%)"), carried
        # through unchanged -- Stage 3 does not re-derive it for table
        # evidence since the header text Stage 2 saw is the same text
        # available here via loc["column"]; re-deriving from the identical
        # input would be redundant, not independent verification.
        elongation_type = pe.get("elongation_type") if pe["property"] == "elongation" else None
        # property_scope/loading_mode: same rationale as elongation_type
        # immediately above -- Stage 2 already classified this from the
        # identical header text Stage 3 would otherwise re-derive from, so
        # it is carried through unchanged rather than recomputed.
        if pe["property"] in ("UTS", "YS"):
            property_scope, loading_mode = pe.get("property_scope"), pe.get("loading_mode")
        else:
            property_scope, loading_mode = None, None
        items.append({"property": pe["property"], "value": value, "source_context": ctx, "unit_info": unit_info,
                       "elongation_type": elongation_type,
                       "property_scope": property_scope, "loading_mode": loading_mode})

    stage3_pst_override = None

    needs_llm = (
        (bool(grade_label) and not any(it["source_context"].get("material_identifier") for it in items))
        or (processing_state_type == "explicit_processing_condition"
            and not any(it["source_context"].get("processing_identifier") for it in items))
        or any(pe["location"]["type"] == "section" for pe in group.get("property_evidence", []))
    )

    # PATCH (post-baseline-run): the bundle is built once here (not inside
    # run_llm_extraction_pass) so its invalid_locations and usable-content
    # check can gate whether the LLM is even called — build_local_bundle
    # itself never crashes on an out-of-range Stage 2 location, but a group
    # whose evidence is entirely invalid/empty (confirmed on dt-2022-10-010's
    # hallucinated table_index and jmrt-2020-06-063's 0-paragraph section)
    # should not spend an LLM call on a content-free prompt.
    bundle = build_local_bundle(paper, material, group)
    for loc in bundle["invalid_locations"]:
        rejected.append({"location": loc, "reason": "invalid_stage2_location",
                          "note": "location does not resolve in the processed paper JSON"})

    if llm_enabled and needs_llm and not bundle_has_usable_content(bundle):
        failures.append({"material_id": material_id, "group_id": group_id,
                          "type": "insufficient_local_evidence",
                          "detail": "no valid local table/section content after filtering invalid locations"})
        needs_llm = False

    if llm_enabled and needs_llm:
        llm_result = run_llm_extraction_pass(paper, material, group, bundle)
        if llm_result["stage3_llm_status"] != "ok":
            failures.append({"material_id": material_id, "group_id": group_id,
                              "type": llm_result["stage3_llm_status"], "detail": llm_result.get("raw_error")})
        else:
            vo = llm_result["verified_output"]
            stage3_pst_override = vo["stage3_processing_state_type"]
            llm_group_identifiers = {
                "material_identifier": vo["verified_material_identifier"],
                "processing_identifier": vo["verified_processing_identifier"],
                "confirmed_unique_specimen_identifiers": derive_confirmed_unique_specimens(paper, vo),
            }

            items = [{**it, "source_context": merge_llm_context_into_source_context(it["source_context"], llm_group_identifiers)}
                     for it in items]

            for cand in vo["verified_prose_properties"]:
                ctx = build_source_context(paper, cand["location"], material_grade_label=grade_label,
                                            processing_state_type=processing_state_type,
                                            group_has_test_condition_evidence=has_test_condition_evidence)
                ctx = merge_llm_context_into_source_context(ctx, llm_group_identifiers)
                # PATCH (v0.4, additive): the model's own reported_unit claim
                # (already present in the existing prompt/verified-output
                # schema) is only trusted if independently confirmed inside
                # the already span-verified source_span -- see
                # resolve_property_unit's docstring.
                unit_info = resolve_property_unit(prose_reported_unit=cand.get("reported_unit"),
                                                    prose_source_span=cand.get("source_span"))
                # Prose path: classify from the model's own reported_property_name
                # text (already part of the existing, unchanged prompt schema --
                # no prompt text was modified for this) plus the verified
                # source_span itself, so a genuine "uniform elongation"/"total
                # elongation" mention in the source prose is still recognized
                # even if the model's reported_property_name paraphrased it.
                elongation_type = None
                if cand["property"] == "elongation":
                    elongation_type = classify_elongation_type(
                        f"{cand.get('reported_property_name', '')} {cand.get('source_span', '')}")
                # property_scope/loading_mode: same treatment as
                # elongation_type immediately above -- classified from the
                # model's own reported_property_name plus the independently
                # span-verified source_span, using the same already-existing
                # prompt schema fields (no prompt text changed).
                property_scope, loading_mode = None, None
                if cand["property"] in ("UTS", "YS"):
                    scope_mode_text = f"{cand.get('reported_property_name', '')} {cand.get('source_span', '')}"
                    property_scope = classify_property_scope(scope_mode_text)
                    loading_mode = classify_loading_mode(scope_mode_text)
                items.append({"property": cand["property"], "value": cand["value"], "source_context": ctx,
                               "unit_info": unit_info, "elongation_type": elongation_type,
                               "property_scope": property_scope, "loading_mode": loading_mode})
            for cand in vo["rejected_prose_properties"]:
                rejected.append({"property": cand.get("property"), "location": cand["location"], "reason": cand["reason"]})

            # Bounded local-discovery candidates without a property value
            # cannot become an observation on their own — preserved for
            # audit visibility, never promoted.
            for disc in vo["verified_discoveries"]:
                rejected.append({"location": disc["location"], "reason": "local_discovery_candidate_not_a_property_value",
                                  "note": disc["reason"], "stage3_discovered": True})

    effective_processing_state_type = stage3_pst_override or processing_state_type
    gate_results = (
        construct_candidates(items, processing_state_type=effective_processing_state_type,
                              material_grade_label=grade_label, n_materials_in_paper=n_materials_in_paper)
        if items else []
    )

    return {
        "material_id": material_id,
        "group_id": group_id,
        "valid_observations": [r for r in gate_results if r["observation_status"] == "valid_observation"],
        "orphan_property_evidence": [r for r in gate_results if r["observation_status"] == "orphan_property_evidence"],
        "rejected_candidates": rejected,
        "failures": failures,
    }


def process_material(paper: dict, material: dict, n_materials_in_paper: int, llm_enabled: bool = False) -> dict:
    composition = None
    for ce in material.get("composition_evidence", []):
        elements = extract_composition_from_table(paper, ce["location"])
        if elements:
            composition = {"status": "reported", "elements": elements}
            break
    if composition is None:
        composition = {"status": "pending_llm_extraction", "elements": []}

    group_results = [
        process_evidence_group(paper, material, g, n_materials_in_paper, llm_enabled=llm_enabled)
        for g in material.get("evidence_groups", [])
    ]

    return {
        "material_id": material["material_id"],
        "reported_grade_label": material.get("grade_label", ""),
        "composition": composition,
        "group_results": group_results,
    }


# ══════════════════════════════════════════════════════════════════════════
# Hierarchical assembly: Paper -> Material -> ProcessingCondition ->
# Observation. ProcessingCondition dedup uses ONLY a positively-matched
# processing identifier (never assumed from absence of conflict). Sharing a
# ProcessingCondition node is NOT observation merging — every valid_observation
# the gate emitted becomes its own, separate Observation entry; two
# observations are never combined post-gate, regardless of how similar they
# look, because merging is exclusively the frozen gate's job, done once, on
# raw evidence items.
# ══════════════════════════════════════════════════════════════════════════

def derive_processing_key(observation: dict) -> str | None:
    """A None key means this observation's processing linkage carries no
    positive, reusable identifier (e.g. confirmed only via the same-row
    fallback) and must never be assumed to share a condition with anything
    else, including another None-key observation."""
    any_prop = next(iter(observation["properties"].values()))
    sc = any_prop["source_context"]
    if sc.get("processing_identifier"):
        return _normalize_for_match(sc["processing_identifier"])
    if sc.get("is_baseline_or_as_received"):
        return "__baseline__"
    return None


def assemble_processing_conditions(material_id: str, tagged_observations: list[tuple[str, dict]]) -> list[dict]:
    """tagged_observations: [(group_id, valid_observation_dict), ...] for one
    material. Groups purely by derive_processing_key(); every observation is
    preserved as its own Observation record — sharing a condition_id never
    combines two observations' properties/replicate structure."""
    conditions: list[dict] = []
    key_to_condition: dict[str, dict] = {}
    for group_id, obs in tagged_observations:
        key = derive_processing_key(obs)
        if key is not None and key in key_to_condition:
            cond = key_to_condition[key]
        else:
            cond = {"condition_id": f"{material_id}-P{len(conditions) + 1}", "processing_key": key, "observations": []}
            conditions.append(cond)
            if key is not None:
                key_to_condition[key] = cond
        obs_out = dict(obs)
        obs_out["observation_id"] = f"{cond['condition_id']}-O{len(cond['observations']) + 1}"
        obs_out["material_id"] = material_id
        obs_out["stage2_evidence_group_ref"] = group_id
        cond["observations"].append(obs_out)
    return conditions


def assemble_paper_record(doi: str, stage2_result: dict, paper: dict, llm_enabled: bool = False) -> dict:
    materials_raw = stage2_result.get("candidate_materials", [])
    n_materials = len(materials_raw)
    materials_processed = [process_material(paper, m, n_materials, llm_enabled=llm_enabled) for m in materials_raw]

    materials_out, unresolved_candidates, rejected_candidates, failures = [], [], [], []

    for mp in materials_processed:
        tagged_observations = []
        for gr in mp["group_results"]:
            tagged_observations += [(gr["group_id"], obs) for obs in gr["valid_observations"]]
            unresolved_candidates += [{"material_id": mp["material_id"], "group_id": gr["group_id"], **orph}
                                       for orph in gr["orphan_property_evidence"]]
            rejected_candidates += [{"material_id": mp["material_id"], "group_id": gr["group_id"], **rej}
                                     for rej in gr["rejected_candidates"]]
            failures += gr["failures"]

        materials_out.append({
            "material_id": mp["material_id"],
            "reported_grade_label": mp["reported_grade_label"],
            "composition": mp["composition"],
            "processing_conditions": assemble_processing_conditions(mp["material_id"], tagged_observations),
        })

    return {
        "doi": doi,
        "stage3_code_version": CODE_VERSION,
        "stage3_prompt_version": PROMPT_VERSION,
        "materials": materials_out,
        "unresolved_candidates": unresolved_candidates,
        "rejected_candidates": rejected_candidates,
        "failures": failures,
    }


def doi_to_processed_filename(doi: str) -> str:
    return doi.replace(".", "-").replace("/", "_") + ".json"
