"""
cpp_prescreen.py — V2 Stage 1: CPP pre-screening.

Paper-level screening of data/processed/*.json, BEFORE any extraction runs.
Frozen design (multiple review rounds, see project memory for full history).
Two independent axes:

  stage1_relevance      — cpp_relevant | non_cpp | ambiguous_relevance | not_assessed
  stage1_extractability — table_extractable | text_extractable | figure_only |
                          mixed_modality | not_extractable_other | no_content | not_assessed

Fields are named `stage1_*` deliberately, not `relevance`/`extractability`: this
stage does NOT get the final word. A future Stage 2 (evidence preparation, with
visibility into full section text and full table rows) is explicitly expected to
be able to REVISE these calls after examining fuller evidence — Stage 1's output
is a triage opinion, not a locked verdict. Any future stage2_relevance /
stage2_reason_code column is meant to sit alongside these, not overwrite them.

Decision order (first match wins):
  0. Deterministic pre-checks (no LLM call):
       - no sections AND no tables            -> not_assessed / skipped_no_content
       - narrow non-steel keyword match (TITLE ONLY, see note below) -> non_cpp / non_steel
       - review/survey title pattern           -> non_cpp / review_or_survey
  1. Otherwise, a structured Qwen call classifies the EXPERIMENTAL DESIGN only
     (material system, what varies, composition/processing status, which
     properties are reported, evidence modality, etc.) — it does NOT extract
     any numbers.
  2-10. Deterministic rules map that structured output to relevance/reason_code.
  11. extractability is derived from evidence_modality, preserving a genuine
     "mixed_modality" outcome rather than collapsing to the best modality.

CORE PRINCIPLE: absence from the limited screening input must NOT be
interpreted as absence from the paper. Concretely:
  - composition_status / processing_status / evidence_modality use the value
    "not_observed", not "missing" — Stage 1 is not claiming the paper lacks
    this, only that it wasn't visible in title+abstract+headings+table
    captions. No rule treats "not_observed" as grounds for exclusion.
  - properties_reported == all false no longer excludes a paper by itself —
    a real experimental paper whose property values simply weren't visible in
    this limited input falls to an explicit ambiguous fallback
    (properties_not_observed_defer_stage2), not a wrong non_cpp exclusion.
  - Exclusion on "no mechanical property" grounds requires a POSITIVE
    characterization (study_nature == microstructure_or_property_
    characterization_only) PLUS independent corroboration: the deterministic
    has_property_header_table signal must NOT contradict it, and the LLM's
    own evidence_sufficiency must not be "low". actamat-2019-11-002 proved
    that even a positive "this paper is about X" claim can be wrong from
    limited input (it looked like a pure crystal-plasticity paper but had a
    real macro-property table) — so this exclusion additionally requires that
    nothing we already independently computed contradicts it. If either
    corroboration fails, the paper is deferred to ambiguous, not excluded.

A SECOND principle applies specifically to variables_changed: a claimed
heat_treatment_processing variation must not, by itself, produce a clean
cpp_relevant decision unless the claim is explicitly supported by something
visible in the Stage 1 input. This is no longer just a self-reported label —
the LLM must additionally quote the specific evidence (a table caption/
heading/row label) into heat_treatment_processing_support_text, and Python
independently checks that quoted text actually contains process-specific
wording (anneal/temper/quench/partition/etc.) before trusting the claim. A
"complete" self-label with no real evidence text, or evidence text that
doesn't contain process wording, is downgraded and does not count. A
"composition varies" claim is separately discounted if material_system_
coherence says single_material, since one material's composition cannot vary
by definition — pure internal-consistency check, no LLM cost. Both discounts
feed a code-side `confirmed_variables` set, which (not the raw
variables_changed list) is what the relevance rules actually use.

has_property_header_table / has_composition_header_table are deterministic
diagnostic signals shown to the LLM as hints, AND used as an independent
corroboration check on rule 4 (see above). They never independently decide
relevance on their own.

Two documented gap-fills, made BEFORE running and not adjusted afterward:
  - confirmed_variables == {"none_single_condition"} is treated as satisfying
    rule 9's "varies composition/processing" requirement, since a genuinely
    single-condition paper (one material, one processing state, reported once)
    is the trivial case of a CPP-relevant paper, not an excluded one.
  - rule 7's "ONLY test/process condition types" requires a NON-EMPTY subset
    of {test_temperature, strain_rate, orientation, shielding_gas,
    other_test_condition} — an empty confirmed_variables set does not
    vacuously satisfy this rule.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROC_DIR = Path("data/processed")
OUT_DIR = Path("data/screening_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-next:80b-a3b-instruct-q4_K_M"

PROMPT_VERSION = "stage1_prescreen_v3"
CODE_VERSION = "cpp_prescreen_v1.3_high_recall_composition_deferral"

LLM_CALL_TIMEOUT_S = 180
LLM_MAX_ATTEMPTS = 2  # one retry on failure (timeout, connection error, bad response), not a bumped timeout

# ── Deterministic keyword lists ────────────────────────────────────────────────
# Word/phrase-boundary matched (see contains_keyword below), not substring
# matched — a prior substring-matching version of this list let "cement" match
# inside "displacement", wrongly excluding real steel papers. "galvanneal" was
# removed entirely: it names a coating/surface-treatment PROCESS applied TO
# steel, not a non-steel material.
#
# matches_non_steel() is applied to the TITLE ONLY (not abstract): a paper
# whose abstract merely *mentions* "aluminum alloy" in a comparison sentence
# is not evidence the paper itself is non-steel, and a title+abstract-wide
# hard-reject was found to risk exactly that false negative. The title is a
# much stronger, safer signal of the paper's actual subject.

NON_STEEL_KW = [
    "zinc cake", "zinc leach",
    "aluminum alloy", "aluminium alloy",
    "titanium alloy", "nickel alloy", "cobalt alloy", "copper alloy",
    "magnesium alloy", "concrete", "geopolymer",
    "fiber reinforced polymer", "carbon fiber", "cfrp",
    "hydrogel", "membrane", "biodegradable",
    "hydrometallurgy", "leaching", "hydroxyapatite",
    "cocrni", "crmnni",
]

# Applied to TITLE only, deliberately narrow. "specification" was removed: a
# standalone word like this is too easily present in a genuine experimental
# steel paper's title (e.g. referencing a material specification standard)
# to be safe as a hard reject on its own.
REVIEW_TITLE_KW = [
    "literature review", "state of the art", "review article",
    "design standard", "design code", "code of practice",
    "standards comparison",
]

PROPERTY_HEADER_KW = [
    "uts", "ts", "ys", "yield", "tensile", "elong", "el (", "el%", "el.",
    "ductility", "strength", "mpa", "%el",
]
COMPOSITION_ELEMENT_KW = {
    "c", "mn", "si", "al", "cr", "ni", "mo", "p", "s", "cu", "v", "nb", "ti", "n", "fe", "w", "co",
}

PROCESSING_TERMS = [
    "anneal", "temper", "quench", "partition", "austenitize", "austenitise",
    "normalize", "normalise", "heat treatment", "heat-treatment", "soak",
    "intercritical", "ia", "q&p", "q-p", "q&t", "austempering", "ausforming",
]

VALID_VARIABLES = {
    "composition", "heat_treatment_processing", "test_temperature", "strain_rate",
    "orientation", "shielding_gas", "other_test_condition", "none_single_condition", "unclear",
}
TEST_CONDITION_VARIABLES = {
    "test_temperature", "strain_rate", "orientation", "shielding_gas", "other_test_condition",
}
VALID_STUDY_NATURE = {
    "original_experimental", "simulation_only", "review_or_survey",
    "mixed_experimental_and_simulation", "microstructure_or_property_characterization_only",
}
VALID_COMPOSITION_STATUS = {"complete", "partial", "nominal_grade_only", "not_observed", "unclear"}
VALID_PROCESSING_STATUS = {"detailed_processing", "baseline_or_nominal_processing", "not_observed", "unclear"}
VALID_COHERENCE = {"single_material", "multiple_related_grades", "multiple_unrelated_materials", "unclear"}
VALID_MODALITY = {"table", "text", "figure_only", "not_observed"}
VALID_SUFFICIENCY = {"high", "medium", "low"}
VALID_PROCESSING_SUPPORT = {"not_claimed", "explicit", "unclear"}

COMPOSITION_OK = {"complete", "partial", "nominal_grade_only"}
PROCESSING_OK = {"detailed_processing", "baseline_or_nominal_processing"}
COHERENCE_MULTI_MATERIAL = {"multiple_related_grades", "multiple_unrelated_materials"}


# ── Deterministic feature helpers ───────────────────────────────────────────────

def contains_keyword(text: str, keywords: list[str]) -> bool:
    """Word/phrase-boundary match, not substring match."""
    text = text.lower()
    for kw in keywords:
        if re.search(r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])", text):
            return True
    return False


def is_no_content(paper: dict) -> bool:
    return not paper.get("sections") and not paper.get("tables")


def matches_non_steel(paper: dict) -> bool:
    return contains_keyword(paper.get("title", ""), NON_STEEL_KW)


def matches_review_title(paper: dict) -> bool:
    return contains_keyword(paper.get("title", ""), REVIEW_TITLE_KW)


def has_property_header_table(paper: dict) -> bool:
    for t in paper.get("tables", []):
        joined = " ".join(h.lower() for h in (t.get("headers") or []))
        if any(kw in joined for kw in PROPERTY_HEADER_KW):
            return True
    return False


def has_composition_header_table(paper: dict) -> bool:
    for t in paper.get("tables", []):
        headers = {h.strip().lower() for h in (t.get("headers") or [])}
        if headers & COMPOSITION_ELEMENT_KW:
            return True
    return False


def has_processing_term(text: str) -> bool:
    """Leading-word-boundary match: catches 'anneal', 'annealing',
    'annealed' via a shared prefix, without matching a term buried mid-word
    (e.g. 'cement' inside 'displacement', 'ia' inside 'material'). No
    trailing boundary requirement, since these words are almost always used
    with a suffix (-ing/-ed) in real papers."""
    text = (text or "").lower()
    for term in PROCESSING_TERMS:
        if re.search(r"(?<![a-z0-9])" + re.escape(term.lower()), text):
            return True
    return False


def build_full_evidence_text(paper: dict) -> str:
    """Everything the model was actually shown: title + abstract + section
    headings + table captions/headers — matches build_prescreen_prompt()'s
    full input exactly. build_evidence_text() alone is too narrow for
    verification: models legitimately quote process terminology from the
    abstract, and checking only headings/captions wrongly rejects a genuine
    quote as unverified."""
    sections_summary, tables_summary = build_evidence_text(paper)
    return f"{paper.get('title', '')} {paper.get('abstract', '')} {sections_summary} {tables_summary}"


def support_text_in_evidence(support_text: str, evidence_text: str) -> bool:
    """Case/whitespace-normalized substring check: does the LLM's claimed
    supporting quote actually appear in the Stage 1 evidence text it was
    given, rather than being a plausible-sounding invention?"""
    norm = lambda s: re.sub(r"\s+", " ", (s or "").lower()).strip()
    needle = norm(support_text)
    return bool(needle) and needle in norm(evidence_text)


# ── Structured LLM screening call ───────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are screening a materials science paper to decide whether it is relevant to a
study of the relationship: steel COMPOSITION -> PROCESSING -> mechanical PROPERTY
(UTS, YS, elongation).

Your job is to CLASSIFY THE EXPERIMENTAL DESIGN of this paper. Do NOT extract any
numerical values, compositions, or property measurements. Only describe what KIND
of study this is.

CRITICAL RULE ON UNCERTAINTY: you are only given the title, abstract, section
HEADINGS, and table CAPTIONS/HEADERS — not full paragraph text or full table rows.
If you cannot tell whether something is present from this limited input, you MUST
say so (use "not_observed" / "unclear" as instructed per field below) rather than
guessing that it is absent. Never treat "I was not shown this" as "this paper does
not have this."

CRITICAL RULE ON heat_treatment_processing: only include "heat_treatment_processing"
in variables_changed if you can point to a SPECIFIC visible table caption, section
heading, or table row/column label that uses process-specific wording (e.g.
anneal, temper, quench, partition, austenitize, normalize, "heat treatment"). A
table or section that only varies a TEMPERATURE VALUE, with no such process
wording attached, is NOT enough evidence — that is very often a TEST temperature,
not a processing temperature. If you set heat_treatment_processing_support to
"explicit", you MUST also quote the specific caption/heading/label you are basing
that on in heat_treatment_processing_support_text. If you cannot quote something
specific, set heat_treatment_processing_support to "unclear" and leave the text
field empty.

CRITICAL RULE ON microstructure_or_property_characterization_only: only use this
study_nature value if you are confident NONE of the visible table captions/headers
or section headings suggest a macro mechanical-property table exists anywhere in
the paper. A paper can look like a microstructure/crystallography study from its
title while still containing one real tensile/property table elsewhere — if you
are not fully confident, prefer "original_experimental" and let properties_reported
reflect what you actually saw instead.

Return ONLY a valid JSON object with this EXACT structure — no extra text, and do
not include the field-name/type descriptions below inside your actual answer. The
example values below are deliberately non-committal placeholders, not a hint about
what the correct answer usually is — replace every one with your own honest answer
for THIS paper:
{
  "material_system": "<name of the material system>",
  "material_system_coherence": "unclear",
  "variables_changed": ["unclear"],
  "heat_treatment_processing_support": "not_claimed",
  "heat_treatment_processing_support_text": "",
  "study_nature": "original_experimental",
  "composition_status": "unclear",
  "processing_status": "unclear",
  "properties_reported": {"UTS": false, "YS": false, "elongation": false},
  "secondary_domain_tags": [],
  "cpp_content_is_incidental": false,
  "evidence_modality": ["not_observed"],
  "evidence_sufficiency": "low",
  "notes": "one sentence justification"
}

Valid options per field:
- material_system_coherence: single_material | multiple_related_grades | multiple_unrelated_materials | unclear
  ("multiple_unrelated_materials" = chemically distinct materials in one study,
  e.g. a dissimilar weld: base metal A + filler + base metal B — NOT the same
  steel family at different grades, which is "multiple_related_grades")
- variables_changed: list any of composition, heat_treatment_processing,
  test_temperature, strain_rate, orientation, shielding_gas, other_test_condition,
  none_single_condition, unclear — list ALL axes that vary, not just the primary one
- heat_treatment_processing_support: not_claimed | explicit | unclear
  (not_claimed if heat_treatment_processing is not in variables_changed at all)
- heat_treatment_processing_support_text: a short direct quote of the caption/
  heading/label supporting an "explicit" claim, or "" if not applicable
- study_nature: original_experimental | simulation_only | review_or_survey |
  mixed_experimental_and_simulation | microstructure_or_property_characterization_only
- composition_status: complete | partial | nominal_grade_only | not_observed | unclear
  ("nominal_grade_only" = a standard grade name like "S690QL" is given without a
  measured composition table; "not_observed" = you were not shown enough to tell,
  NOT a claim that the paper lacks composition data)
- processing_status: detailed_processing | baseline_or_nominal_processing | not_observed | unclear
- properties_reported: true means observed in what you were shown; false means
  not observed in what you were shown — NOT a claim that the paper lacks it
- secondary_domain_tags: list other topics (corrosion, coating, welding, fatigue,
  fracture, etc.) even if the paper is fundamentally a CPP study
- cpp_content_is_incidental: true only if composition/processing/property content
  is a passing mention, not a substantive part of the paper
- evidence_modality: list every way properties are reported that you can see —
  table, text, figure_only, not_observed. If some values are in a table and
  others only in a figure, include both "table" and "figure_only"
- evidence_sufficiency: your own honest confidence (high/medium/low) in the
  judgments above, given how much of the paper you were actually shown
"""


def build_evidence_text(paper: dict) -> tuple[str, str]:
    """Returns (sections_summary, tables_summary) — the Stage 1 visible
    evidence: table captions/headers and section headings only. Used both to
    build the prompt below AND to verify that a claimed
    heat_treatment_processing_support_text quote actually came from this same
    text, not from the model's own invention."""
    tables_summary = ""
    for i, t in enumerate(paper.get("tables", []), 1):
        tables_summary += f"\nTable {i}: {t.get('caption', '')}\n"
        if t.get("headers"):
            tables_summary += "  Headers: " + " | ".join(t["headers"]) + "\n"

    sections_summary = ""
    for s in paper.get("sections", []):
        h = s.get("heading", "")
        if h:
            sections_summary += f"  - {h}\n"

    return sections_summary, tables_summary


def build_prescreen_prompt(paper: dict, det_hints: dict) -> str:
    sections_summary, tables_summary = build_evidence_text(paper)

    return f"""{SYSTEM_PROMPT}

TITLE: {paper.get('title', '')}

ABSTRACT:
{paper.get('abstract', '')}

SECTION HEADINGS:
{sections_summary if sections_summary else '(none available)'}

TABLE CAPTIONS AND HEADERS (not full data):
{tables_summary if tables_summary else '(none available)'}

Deterministic hints (may be imperfect, use as context only):
  n_tables={det_hints['n_tables']}, n_sections={det_hints['n_sections']},
  has_property_header_table={det_hints['has_property_header_table']},
  has_composition_header_table={det_hints['has_composition_header_table']}
"""


class LLMCallError(Exception):
    pass


def call_llm(prompt: str) -> str:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096},
        }, timeout=LLM_CALL_TIMEOUT_S)
    except requests.exceptions.RequestException as e:
        raise LLMCallError(f"request failed: {type(e).__name__}: {e}")

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


def call_llm_with_retry(prompt: str, max_attempts: int = LLM_MAX_ATTEMPTS) -> tuple[str, str]:
    """Single controlled retry, no timeout change. Returns (response_text,
    attempt_log_string). Raises LLMCallError only if every attempt fails, with
    the combined per-attempt diagnostic log included in the exception message
    so a failure is at least explainable after the fact rather than opaque."""
    attempt_log = []
    last_error = None
    for attempt in range(1, max_attempts + 1):
        t_attempt = time.time()
        try:
            text = call_llm(prompt)
            attempt_log.append(f"attempt{attempt}: ok in {time.time() - t_attempt:.1f}s")
            return text, " | ".join(attempt_log)
        except LLMCallError as e:
            elapsed = time.time() - t_attempt
            attempt_log.append(f"attempt{attempt}: failed after {elapsed:.1f}s ({e})")
            last_error = e
    raise LLMCallError(f"all {max_attempts} attempts failed: " + " | ".join(attempt_log)) from last_error


def parse_output(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ── Strict parsing helpers ──────────────────────────────────────────────────────

def strict_bool(value, default: bool = False) -> bool:
    """Only an actual JSON boolean, or the exact strings 'true'/'false'
    (case-insensitive), are accepted. Anything else (including other truthy-
    looking strings) falls back to `default` rather than being coerced by
    Python's normal truthiness rules — a raw `bool("false")` is True in
    Python, which would silently corrupt this field if not guarded against."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
    return default


def normalize_enum(value, valid_set: set[str], default: str) -> str:
    if isinstance(value, str):
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        if v in valid_set:
            return v
    return default


def normalize_llm_output(raw: dict) -> dict:
    """Defensive normalisation: unknown/missing/malformed values fall back to
    the most conservative option for that field (never silently reinterpreted
    as a favourable value)."""
    out = {}
    out["material_system"] = raw.get("material_system") if isinstance(raw.get("material_system"), str) else "unclear"
    if not out["material_system"]:
        out["material_system"] = "unclear"

    out["material_system_coherence"] = normalize_enum(
        raw.get("material_system_coherence"), VALID_COHERENCE, "unclear")

    variables = raw.get("variables_changed")
    if isinstance(variables, list):
        cleaned = []
        for v in variables:
            if isinstance(v, str) and v.strip().lower().replace(" ", "_") in VALID_VARIABLES:
                cleaned.append(v.strip().lower().replace(" ", "_"))
        out["variables_changed"] = cleaned
    else:
        out["variables_changed"] = []

    out["heat_treatment_processing_support"] = normalize_enum(
        raw.get("heat_treatment_processing_support"), VALID_PROCESSING_SUPPORT, "unclear")
    support_text = raw.get("heat_treatment_processing_support_text")
    out["heat_treatment_processing_support_text"] = support_text if isinstance(support_text, str) else ""

    out["study_nature"] = normalize_enum(raw.get("study_nature"), VALID_STUDY_NATURE, "unclear")
    out["composition_status"] = normalize_enum(raw.get("composition_status"), VALID_COMPOSITION_STATUS, "unclear")
    out["processing_status"] = normalize_enum(raw.get("processing_status"), VALID_PROCESSING_STATUS, "unclear")

    props = raw.get("properties_reported")
    if isinstance(props, dict):
        out["properties_reported"] = {
            "UTS": strict_bool(props.get("UTS"), False),
            "YS": strict_bool(props.get("YS"), False),
            "elongation": strict_bool(props.get("elongation"), False),
        }
    else:
        out["properties_reported"] = {"UTS": False, "YS": False, "elongation": False}

    tags = raw.get("secondary_domain_tags")
    out["secondary_domain_tags"] = (
        [str(t).strip().lower() for t in tags if isinstance(t, str)] if isinstance(tags, list) else []
    )

    out["cpp_content_is_incidental"] = strict_bool(raw.get("cpp_content_is_incidental"), False)

    modality = raw.get("evidence_modality")
    if isinstance(modality, list):
        cleaned = []
        for m in modality:
            if isinstance(m, str) and m.strip().lower().replace(" ", "_") in VALID_MODALITY:
                cleaned.append(m.strip().lower().replace(" ", "_"))
        out["evidence_modality"] = cleaned
    else:
        out["evidence_modality"] = []

    out["evidence_sufficiency"] = normalize_enum(raw.get("evidence_sufficiency"), VALID_SUFFICIENCY, "low")
    out["notes"] = raw.get("notes") if isinstance(raw.get("notes"), str) else ""
    return out


def compute_confirmed_variables(llm: dict, evidence_text: str) -> set[str]:
    """Discount unsupported composition/heat_treatment_processing claims.

    'composition' is only logically coherent if material_system_coherence
    indicates more than one material — a single material cannot have varying
    composition by definition.

    'heat_treatment_processing' is only trusted if the LLM (a) self-labelled
    it "explicit" AND (b) actually quoted supporting text AND (c) that quoted
    text genuinely contains process-specific wording AND (d) that quoted text
    is actually present in the Stage 1 evidence text the model was given —
    (c) alone only checks the quote's content, not its provenance, so a
    fabricated but plausible-sounding quote would otherwise still pass. The
    self-label alone is never trusted — this is deliberately not "the model
    grading its own claim," it's an independent check on evidence the model
    was required to produce and that we can verify against what it actually saw."""
    confirmed = set()
    for v in llm["variables_changed"]:
        if v == "composition":
            if llm["material_system_coherence"] in COHERENCE_MULTI_MATERIAL:
                confirmed.add(v)
        elif v == "heat_treatment_processing":
            support = llm["heat_treatment_processing_support"]
            support_text = llm["heat_treatment_processing_support_text"]
            if (support == "explicit" and support_text
                    and has_processing_term(support_text)
                    and support_text_in_evidence(support_text, evidence_text)):
                confirmed.add(v)
        else:
            confirmed.add(v)
    return confirmed


# ── Deterministic decision logic ────────────────────────────────────────────────

def apply_deterministic_prechecks(paper: dict) -> dict | None:
    """Returns a finished screening record if a deterministic rule resolves the
    paper without needing an LLM call, else None."""
    if is_no_content(paper):
        return {
            "stage1_relevance": "not_assessed", "stage1_reason_code": "skipped_no_content",
            "stage1_extractability": "no_content", "decision_basis": "deterministic",
            "evidence_sufficiency": "high", "rule_fired": "det_no_content",
        }
    if matches_non_steel(paper):
        return {
            "stage1_relevance": "non_cpp", "stage1_reason_code": "non_steel",
            "stage1_extractability": "not_assessed", "decision_basis": "deterministic",
            "evidence_sufficiency": "high", "rule_fired": "det_non_steel",
        }
    if matches_review_title(paper):
        return {
            "stage1_relevance": "non_cpp", "stage1_reason_code": "review_or_survey",
            "stage1_extractability": "not_assessed", "decision_basis": "deterministic",
            "evidence_sufficiency": "high", "rule_fired": "det_review_title",
        }
    return None


def apply_llm_decision_rules(llm: dict, confirmed_variables: set[str], det_has_property_header_table: bool) -> tuple[str, str, str]:
    """Returns (relevance, reason_code, rule_fired)."""
    props = llm["properties_reported"]
    any_property = props["UTS"] or props["YS"] or props["elongation"]

    # step 2
    if llm["study_nature"] == "simulation_only":
        return "non_cpp", "simulation_only", "llm_simulation_only"
    # step 3
    if llm["study_nature"] == "review_or_survey":
        return "non_cpp", "review_or_survey", "llm_review_or_survey"
    # step 4 — positive characterization required, AND independent corroboration
    # required (deterministic table-header scan must not contradict it, and the
    # LLM's own confidence must not already be low) before this ever excludes.
    if llm["study_nature"] == "microstructure_or_property_characterization_only":
        if (not det_has_property_header_table) and llm["evidence_sufficiency"] != "low":
            return "non_cpp", "microstructure_only_no_mechanical_target", "llm_no_mechanical_property"
        return "ambiguous_relevance", "microstructure_claim_needs_stage2_verification", "llm_microstructure_claim_unconfirmed"
    # step 5
    corrosion_or_coating = bool({"corrosion", "coating"} & set(llm["secondary_domain_tags"]))
    if corrosion_or_coating and llm["cpp_content_is_incidental"]:
        return "non_cpp", "corrosion_or_coating_primary_focus", "llm_corrosion_coating_primary"
    # step 6
    if llm["material_system_coherence"] == "multiple_unrelated_materials":
        within_material_var = bool({"composition", "heat_treatment_processing"} & confirmed_variables)
        if within_material_var and any_property:
            return "ambiguous_relevance", "multi_material_needs_disaggregation", "llm_multi_material_partial"
        return "non_cpp", "nonstandard_independent_variable_excluded", "llm_multi_material_excluded"
    # step 7 — processing_status == "unclear" is treated the same as PROCESSING_OK
    # here (not elsewhere): confirmed false rejections of commatsci-2025-114088
    # and jcsr-2020-106348 both had composition_status acceptable and only
    # processing_status=="unclear" standing between them and retention. A
    # test-condition-only paper with a plausible composition and an unclear
    # (not confirmed-absent) processing state should not be hard-excluded —
    # Stage 2 can resolve whether a usable baseline observation exists.
    test_only = len(confirmed_variables) > 0 and confirmed_variables.issubset(TEST_CONDITION_VARIABLES)
    if test_only:
        if llm["composition_status"] in COMPOSITION_OK and (
            llm["processing_status"] in PROCESSING_OK or llm["processing_status"] == "unclear"
        ):
            return "ambiguous_relevance", "nonstandard_independent_variable_partial", "llm_nonstandard_variable_partial"
        # PATCH (composition-not-observed asymmetry fix): composition_status
        # == "not_observed" means composition simply wasn't VISIBLE in
        # Stage 1's limited input (title/abstract/headings/captions), not
        # that it is positively absent or incompatible — the identical
        # principle step 8 below already applies to the general
        # (non-test-only) case. Without this, a test-condition-only paper
        # whose composition is only stated in full prose was hard-excluded
        # here even though nothing actually contradicts CPP relevance.
        # Mirrors step 8's own reason code (deliberately, not inventing a
        # new one) and its any_property gate. composition_status ==
        # "unclear" (the model WAS shown something and is genuinely
        # uncertain) is NOT covered by this — that remains a hard exclusion,
        # unchanged, since it is a materially different, weaker signal than
        # "not shown enough to tell".
        if (llm["composition_status"] == "not_observed"
                and (llm["processing_status"] in PROCESSING_OK or llm["processing_status"] == "unclear")
                and any_property):
            return "ambiguous_relevance", "composition_not_observed_otherwise_relevant", "llm_test_only_composition_not_observed"
        # PATCH (composition-unclear deferral fix): composition_status ==
        # "unclear" has no field-specific stronger-negative meaning in the
        # prompt (only "not_observed" is explicitly defined there), and is
        # ALSO normalize_enum()'s fallback for a malformed/missing raw
        # value — a parsing artifact carrying no information about the
        # paper at all. A whole-record LLM call/parse failure already
        # defers the entire record to ambiguous_relevance elsewhere in this
        # module (see screen_paper's llm_call_failed/llm_parse_failed
        # paths); a single field falling back the same way should not be
        # treated more harshly. Mirrors the not_observed branch immediately
        # above exactly, with its own distinct reason_code/rule_fired so
        # the two source states stay distinguishable in the audit trail —
        # composition_status itself is never rewritten to "not_observed".
        if (llm["composition_status"] == "unclear"
                and (llm["processing_status"] in PROCESSING_OK or llm["processing_status"] == "unclear")
                and any_property):
            return "ambiguous_relevance", "composition_unclear_otherwise_relevant", "llm_test_only_composition_unclear"
        return "non_cpp", "nonstandard_independent_variable_excluded", "llm_nonstandard_variable_excluded"
    # step 8 — composition not observed, everything else present -> ambiguous, never excluded
    if (llm["composition_status"] == "not_observed"
            and llm["processing_status"] in PROCESSING_OK
            and any_property):
        return "ambiguous_relevance", "composition_not_observed_otherwise_relevant", "llm_composition_not_observed"
    # PATCH (composition-unclear deferral fix): same rationale as the
    # test-only branch's equivalent patch above — "unclear" is not a
    # documented stronger-negative signal than "not_observed" for this
    # field, and is also normalize_enum()'s malformed/missing-value
    # fallback. Without this, an otherwise-relevant paper with an
    # "unclear" composition_status still ends up at ambiguous_relevance via
    # the step-10 catch-all below, but with the generic, uninformative
    # "unclassified_needs_review" reason — this gives it the same specific,
    # accurate reason code step 8 already gives the not_observed case.
    if (llm["composition_status"] == "unclear"
            and llm["processing_status"] in PROCESSING_OK
            and any_property):
        return "ambiguous_relevance", "composition_unclear_otherwise_relevant", "llm_composition_unclear"
    # step 9 (includes the documented none_single_condition gap-fill)
    varies_composition_or_processing = bool({"composition", "heat_treatment_processing"} & confirmed_variables)
    is_single_condition = confirmed_variables == {"none_single_condition"}
    if ((varies_composition_or_processing or is_single_condition)
            and llm["composition_status"] in COMPOSITION_OK
            and llm["processing_status"] in PROCESSING_OK
            and any_property):
        return "cpp_relevant", "keep_cpp_relevant", "llm_keep_cpp_relevant"
    # step 9b — looks like a real experimental paper, but properties simply were
    # not visible in this limited input: defer, do not exclude and do not confirm
    if llm["study_nature"] in ("original_experimental", "mixed_experimental_and_simulation") and not any_property:
        return "ambiguous_relevance", "properties_not_observed_defer_stage2", "llm_properties_not_observed"
    # step 10
    return "ambiguous_relevance", "unclassified_needs_review", "llm_unclassified_needs_review"


MODALITY_MAP = {"table": "table_extractable", "text": "text_extractable", "figure_only": "figure_only"}


def derive_extractability(relevance: str, evidence_modality: list[str]) -> str:
    if relevance not in ("cpp_relevant", "ambiguous_relevance"):
        return "not_assessed"
    modality_set = set(evidence_modality) - {"not_observed"}
    if not modality_set:
        return "not_extractable_other"
    if len(modality_set) == 1:
        return MODALITY_MAP[next(iter(modality_set))]
    return "mixed_modality"


# ── Main per-paper screening ────────────────────────────────────────────────────

_EMPTY_LLM_FIELDS = {
    "material_system": None, "material_system_coherence": None, "variables_changed": None,
    "confirmed_variables": None, "heat_treatment_processing_support": None,
    "heat_treatment_processing_support_text": None, "heat_treatment_support_verified_in_source": None,
    "study_nature": None, "composition_status": None, "processing_status": None,
    "properties_reported_UTS": None, "properties_reported_YS": None, "properties_reported_elongation": None,
    "secondary_domain_tags": None, "cpp_content_is_incidental": None,
    "evidence_modality": None, "notes": None,
    "llm_parse_ok": None, "llm_attempt_log": None,
}


def _finish_record(record: dict, t0: float) -> dict:
    record["runtime_s"] = round(time.time() - t0, 2)
    record["screening_model"] = record.get("screening_model", MODEL)
    record["prompt_version"] = PROMPT_VERSION
    record["code_version"] = CODE_VERSION
    record["screened_at"] = datetime.now(timezone.utc).isoformat()
    return record


def screen_paper(pf: Path) -> dict:
    t0 = time.time()

    try:
        paper = json.loads(pf.read_text())
    except (OSError, json.JSONDecodeError) as e:
        record = {
            "doi": pf.stem, "title": "", "journal": "", "year": "",
            "n_tables": None, "n_sections": None,
            "has_property_header_table": None, "has_composition_header_table": None,
            "stage1_relevance": "not_assessed", "stage1_reason_code": "paper_read_error",
            "stage1_extractability": "not_assessed", "decision_basis": "deterministic",
            "evidence_sufficiency": "low", "rule_fired": "det_paper_read_error",
            "screening_model": "n/a",
        }
        record.update(_EMPTY_LLM_FIELDS)
        record["notes"] = f"{type(e).__name__}: {e}"
        return _finish_record(record, t0)

    doi = paper.get("doi", pf.stem)
    record = {
        "doi": doi,
        "title": paper.get("title", ""),
        "journal": paper.get("journal", ""),
        "year": paper.get("year", ""),
        "n_tables": len(paper.get("tables", [])),
        "n_sections": len(paper.get("sections", [])),
        "has_property_header_table": has_property_header_table(paper),
        "has_composition_header_table": has_composition_header_table(paper),
    }

    det = apply_deterministic_prechecks(paper)
    if det is not None:
        record.update(det)
        record.update(_EMPTY_LLM_FIELDS)
        record["screening_model"] = "deterministic_only"
        return _finish_record(record, t0)

    det_hints = {
        "n_tables": record["n_tables"], "n_sections": record["n_sections"],
        "has_property_header_table": record["has_property_header_table"],
        "has_composition_header_table": record["has_composition_header_table"],
    }
    prompt = build_prescreen_prompt(paper, det_hints)

    try:
        raw_text, attempt_log = call_llm_with_retry(prompt)
        raw_json = parse_output(raw_text)
    except LLMCallError as e:
        record.update({
            "stage1_relevance": "ambiguous_relevance", "stage1_reason_code": "llm_call_failed",
            "stage1_extractability": "not_assessed", "decision_basis": "llm_structured",
            "evidence_sufficiency": "low", "rule_fired": "llm_call_error",
        })
        record.update(_EMPTY_LLM_FIELDS)
        record["notes"] = str(e)
        record["llm_parse_ok"] = False
        record["llm_attempt_log"] = str(e)
        record["screening_model"] = MODEL
        return _finish_record(record, t0)

    if raw_json is None:
        record.update({
            "stage1_relevance": "ambiguous_relevance", "stage1_reason_code": "llm_parse_failed",
            "stage1_extractability": "not_assessed", "decision_basis": "llm_structured",
            "evidence_sufficiency": "low", "rule_fired": "llm_parse_error",
        })
        record.update(_EMPTY_LLM_FIELDS)
        record["notes"] = raw_text[:300]
        record["llm_parse_ok"] = False
        record["llm_attempt_log"] = attempt_log
        record["screening_model"] = MODEL
        return _finish_record(record, t0)

    llm = normalize_llm_output(raw_json)
    evidence_text = build_full_evidence_text(paper)
    heat_treatment_support_verified_in_source = (
        support_text_in_evidence(llm["heat_treatment_processing_support_text"], evidence_text)
        if llm["heat_treatment_processing_support"] == "explicit" else False
    )
    confirmed_variables = compute_confirmed_variables(llm, evidence_text)
    relevance, reason_code, rule_fired = apply_llm_decision_rules(
        llm, confirmed_variables, record["has_property_header_table"])
    extractability = derive_extractability(relevance, llm["evidence_modality"])

    record.update({
        "stage1_relevance": relevance, "stage1_reason_code": reason_code,
        "stage1_extractability": extractability, "decision_basis": "llm_structured",
        "evidence_sufficiency": llm["evidence_sufficiency"], "rule_fired": rule_fired,
        "material_system": llm["material_system"],
        "material_system_coherence": llm["material_system_coherence"],
        "variables_changed": json.dumps(llm["variables_changed"]),
        "confirmed_variables": json.dumps(sorted(confirmed_variables)),
        "heat_treatment_processing_support": llm["heat_treatment_processing_support"],
        "heat_treatment_processing_support_text": llm["heat_treatment_processing_support_text"],
        "heat_treatment_support_verified_in_source": heat_treatment_support_verified_in_source,
        "study_nature": llm["study_nature"],
        "composition_status": llm["composition_status"],
        "processing_status": llm["processing_status"],
        "properties_reported_UTS": llm["properties_reported"]["UTS"],
        "properties_reported_YS": llm["properties_reported"]["YS"],
        "properties_reported_elongation": llm["properties_reported"]["elongation"],
        "secondary_domain_tags": json.dumps(llm["secondary_domain_tags"]),
        "cpp_content_is_incidental": llm["cpp_content_is_incidental"],
        "evidence_modality": json.dumps(llm["evidence_modality"]),
        "notes": llm["notes"],
        "llm_parse_ok": True,
        "llm_attempt_log": attempt_log,
    })
    record["screening_model"] = MODEL
    return _finish_record(record, t0)


COMBINED_STATUS_MAP_EXTRACTABLE = {"table_extractable", "text_extractable", "mixed_modality"}


def combined_status(relevance: str, extractability: str) -> str:
    if relevance == "non_cpp":
        return "non_cpp"
    if relevance == "not_assessed":
        return "not_assessed"
    if relevance == "ambiguous_relevance":
        return "ambiguous"
    if relevance == "cpp_relevant":
        if extractability in COMBINED_STATUS_MAP_EXTRACTABLE:
            return "cpp_extractable"
        return "cpp_relevant_but_not_extractable"
    return "unclassified"


def main(doi_list_path: Path | None = None, out_name: str = "paper_classification_v2_test17.csv"):
    if doi_list_path is not None:
        stems = [line.strip() for line in doi_list_path.read_text().splitlines() if line.strip()]
        files = [PROC_DIR / f"{s}.json" for s in stems]
    else:
        files = sorted(PROC_DIR.glob("*.json"))

    records = []
    n_deterministic = 0
    n_llm = 0
    for i, pf in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {pf.stem}")
        rec = screen_paper(pf)
        rec["stage1_combined_status"] = combined_status(rec["stage1_relevance"], rec["stage1_extractability"])
        records.append(rec)
        if rec["decision_basis"] == "deterministic":
            n_deterministic += 1
        else:
            n_llm += 1
        print(f"  stage1_relevance={rec['stage1_relevance']}  reason={rec['stage1_reason_code']}  "
              f"extractability={rec['stage1_extractability']}  rule={rec['rule_fired']}  "
              f"({rec['runtime_s']}s)")

    df = pd.DataFrame(records)
    out_path = OUT_DIR / out_name
    df.to_csv(out_path, index=False)

    print("=" * 60)
    print(f"Screened {len(records)} papers")
    print(f"Deterministic-only decisions: {n_deterministic}")
    print(f"LLM-assisted decisions: {n_llm}")
    print(f"Output: {out_path}")
    return df


if __name__ == "__main__":
    main()
