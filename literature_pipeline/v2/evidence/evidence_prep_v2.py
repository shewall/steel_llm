"""
evidence_prep_v2.py — V2 Stage 2, second implementation (v2), alongside the
FROZEN v1 (literature_pipeline/v2/evidence/evidence_prep.py — not modified,
not imported from; this module is fully self-contained, same convention as
every other stage in this pipeline).

── Why v2 exists ─────────────────────────────────────────────────────────────
The frozen Stage 3 validation runs showed Stage 3's safety gate correctly
rejecting bad Stage 2 v1 evidence far more often than it accepted good
evidence — the recall bottleneck is upstream, in how Stage 2 v1 discovers
and cites evidence, not in Stage 3's linkage safety. Confirmed Stage 2 v1
failure modes this version targets:
  - a total LLM-call failure could reduce a whole paper to zero materials/
    groups (jalmes-2025-100214);
  - a paper-level material could be silently missed entirely if it never
    appeared in whatever the LLM happened to attend to (heliyon's DP1-DP3
    never surfaced, only DP4 did);
  - a single evidence_group could combine evidence from genuinely different
    specimen histories (heliyon's standard-tensile UTS + RPS-pre-strain YS);
  - a citation could reference a location that plainly doesn't exist in the
    source at all (dt-2022-10-010's table_index=6 in a 3-table paper);
  - a citation could reference a real table but the wrong row/column
    (ijmecsci-2025-110565);
  - a citation could paraphrase instead of pointing at grounded text, or
    point at a structurally empty section (jmrt-2020-06-063, actamat's
    Q&P sentence missed alongside its Q&T sibling).

── Core architectural fix ────────────────────────────────────────────────────
The LLM is never asked to invent a source location. A deterministic Layer A
first enumerates every real, stable candidate location in the processed
paper JSON (a "candidate inventory") and assigns each one a stable ID. The
LLM (Layer B) may only select, classify, and group these IDs — it can never
author a new table_index/row_index/column/paragraph_index. Layer C then
audits Layer B's output against the full inventory: every candidate ID that
carries a target signal must land in exactly one of four states —
`assigned`, `known_non_target`, `unresolved`, or (only if it landed in none
of the first three) `unaccounted`. A single bounded recovery pass targets
only genuinely unaccounted candidates; anything still unaccounted after that
is deterministically, explicitly converted to `unresolved_evidence` by
Python (never silently dropped, never left ambiguous with an LLM-explicit
unresolved judgement — the two are tagged with different `source` values).

── The four approved clarifications, as implemented ─────────────────────────
1. Every table_cell inventory record carries `source_column` (the exact raw
   header string) and `raw_value` (the exact raw cell token) alongside its
   `canonical_alias` — the alias is metadata, a lookup key for the LLM to
   reason with; it never replaces or overwrites the raw representation.
2. Material discovery (Call 1) is built from the union of every
   signal-bearing candidate regardless of which signal fired — a grade
   label sitting only in a property table row or a processing paragraph is
   just as visible to material discovery as one sitting in a composition
   table.
3. `unaccounted` (never examined by any LLM output at all) and `unresolved`
   (explicitly examined and explicitly not confidently assignable) are
   computed and reported as two separate rates, never conflated.
4. The candidate inventory (Layer A) is immutable once built — it contains
   no LLM-derived fields, and nothing in Layer B or C ever writes into it.
   Layer B's classification lives in entirely separate return structures
   that only ever *reference* candidate IDs.

── What this module does NOT do ──────────────────────────────────────────────
  - No paper-specific keywords or rules for any of the six known-difficult
    papers — every heuristic here is a general steel/materials-science
    evidence signal (grade-label shape, standard property symbols like
    Rm/Rp0.2, standard processing/test-condition terms, standard composition
    element symbols), not tuned to any one paper's exact wording.
  - No numerical extraction, no unit conversion, no final CPP observations —
    identical scope boundary to v1; this is still evidence *preparation*.
  - No modification of Stage 1, Stage 2 v1, Stage 3, or any legacy file.
  - No real LLM calls in this round. The LLM call scaffold (self-contained
    copy of Stage 3's now-corrected caller-side hard-timeout mechanism) is
    present so the module compiles end-to-end and a future round can wire
    it up, but process_paper_v2() only ever runs against *injected* call
    functions in this file's own test suite — never against Ollama.

── Not run yet ───────────────────────────────────────────────────────────────
Implementation and offline testing only. main() is deliberately NOT
implemented for the same reason Stage 3's wasn't at this point: prompt
content for the three LLM call types has not been designed/reviewed with
the same rigor as every other stage's prompt was before being trusted, and
no orchestration against the real corpus should be possible until that
happens.
"""

import json
import re
import threading
from pathlib import Path

import requests

PROC_DIR = Path("data/processed")
OUT_DIR = Path("data/evidence_v2/stage2_v2_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-next:80b-a3b-instruct-q4_K_M"

CODE_VERSION = "evidence_prep_v2_v0.7_column_based_material_label_accounting"
PROMPT_VERSION = "stage2_v2_prompts_v1"  # actual text defined near call_llm_with_retry, below

LLM_CALL_TIMEOUT_S = 240
LLM_WALLCLOCK_HARD_TIMEOUT_S = 260
LLM_MAX_ATTEMPTS = 2


# ══════════════════════════════════════════════════════════════════════════
# General, non-paper-specific materials-science evidence heuristics.
# Deliberately broad — false positives here are acceptable (Layer B/C
# classify and account for them); silently missing evidence is not.
# ══════════════════════════════════════════════════════════════════════════

def _normalize_for_match(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower())
    s = re.sub(r"\s*([%°±])\s*", r"\1", s)
    return s.strip()


ELEMENT_SYMBOLS = {
    "C", "MN", "SI", "AL", "CR", "NI", "MO", "P", "S", "CU", "V", "NB", "TI",
    "N", "FE", "W", "CO", "B", "ZR", "SN", "CA", "O", "H", "AS",
}

# Standard property symbols/aliases across common table-header conventions
# (Rm/Rp0.2 are the standard EN/ISO tensile-strength/yield-strength symbols,
# not specific to any one paper). Checked in this order so "elongation"
# terms are matched before shorter, more ambiguous UTS/YS abbreviations.
# PATCH (bare rm/ys substring fix): the bare tokens "rm" and "ys" formerly
# lived in this dict and were matched via plain substring `in` checks --
# they false-positived inside unrelated words ("atheRMal", "defoRMation",
# "cRYStallized", "sYStem", "catalYSt"; verified: 180 distinct corpus header
# strings, 139/870 papers). They are matched instead below via
# word-boundary-anchored regexes, the same treatment already given to the
# elongation bare abbreviations (UE/TE/EL) for the identical reason.
PROPERTY_ALIAS_TERMS = {
    "elongation": ["elongation", "el%", "%el", "el.", " el ", "a50", "a_50", "ag", "a_g", "uniform elong", "total elong"],
    "YS": ["yield strength", "yield stress", "rp0.2", "rp 0.2", "r_p0.2", "proof stress", "upper yield", "lower yield"],
    "UTS": ["ultimate tensile", "tensile strength", "tensile stress", "uts", " ts ", "ts (", "ts(", "ultimate strength"],
}

# Bare short-form elongation abbreviations (UE, TE, EL, standalone "A%")
# cannot safely be added to PROPERTY_ALIAS_TERMS above -- that list is
# matched with plain substring `in` checks, and 2-letter substrings like
# "ue"/"te" would false-positive inside ordinary words ("value", "steel",
# "temperature"). These are matched instead with word-boundary-anchored
# regexes against the ORIGINAL (non-lowercased-substring-squeezed) header
# text, so "UE [%]"/"TE [%]" match but "Value (mm)"/"Temperature (°C)"
# never do.
_ELONGATION_BARE_ABBREV_RE = re.compile(r"\b(?:UE|TE|EL)\b|\bA\s?g\b|\bA\s?%", re.IGNORECASE)


def _is_bare_elongation_abbreviation(header: str) -> bool:
    return bool(header) and bool(_ELONGATION_BARE_ABBREV_RE.search(header))


# Bare UTS/YS symbol-style headers (Rm, YS) -- same rationale and treatment
# as the elongation bare abbreviations immediately above: "rm" and "ys" as
# PLAIN substrings false-positive inside unrelated words ("atheRMal",
# "defoRMation", "cRYStallized", "sYStem", "catalYSt" -- all real,
# corpus-observed false matches; see docs/thesis_research_log_20260913.md
# §3.5). Word-boundary anchoring against the ORIGINAL header text fixes
# this. "R[\s_]?m" (not a plain "\bRm\b") additionally covers a real,
# corpus-observed rendering where a stripped subscript leaves a literal
# space between "R" and "m" (e.g. "R m [MPa]") -- without this, that
# genuine header would go unrecognized entirely (it was never matched by
# the old plain-substring "rm" check either, which required "r" and "m"
# adjacent with no space).
_UTS_BARE_RM_RE = re.compile(r"\bR[\s_]?m\b", re.IGNORECASE)
_YS_BARE_ABBREV_RE = re.compile(r"\bYS\b", re.IGNORECASE)


def _is_bare_uts_symbol(header: str) -> bool:
    return bool(header) and bool(_UTS_BARE_RM_RE.search(header))


def _is_bare_ys_abbreviation(header: str) -> bool:
    return bool(header) and bool(_YS_BARE_ABBREV_RE.search(header))


# Elongation sub-type classification -- conservative, source-wording-only
# (never inferred from numeric magnitude or table position). Order matters:
# fracture-family wording is checked first since "elongation at break"
# would otherwise also match the bare "elongation" term.
_ELONGATION_FRACTURE_RE = re.compile(
    r"elongation\s*(?:at|to)\s*(?:break|failure)|elongation\s+after\s+(?:break|fracture)|fracture\s+elongation",
    re.IGNORECASE)
_ELONGATION_TOTAL_RE = re.compile(r"total\s+elong|\bTE\b", re.IGNORECASE)
_ELONGATION_UNIFORM_RE = re.compile(r"uniform\s+elong|\bUE\b|\bA\s?g\b|\bA_g\b", re.IGNORECASE)


def classify_elongation_type(text: str) -> str:
    """Conservative, deterministic elongation sub-type classification from
    source wording alone -- 'total' | 'uniform' | 'fracture' |
    'unspecified'. Never infers a type from numeric magnitude or table
    position; a bare "Elongation (%)" with no qualifying wording is
    'unspecified', never silently promoted to 'total'. Checked in
    fracture -> total -> uniform order so more specific fracture-family
    wording is never shadowed by the bare "elongation" substring."""
    text = text or ""
    if _ELONGATION_FRACTURE_RE.search(text):
        return "fracture"
    if _ELONGATION_TOTAL_RE.search(text):
        return "total"
    if _ELONGATION_UNIFORM_RE.search(text):
        return "uniform"
    return "unspecified"


# Property scope (measurement level) and loading mode (tensile vs
# compressive) -- conservative, source-wording-only, from the column
# HEADER TEXT ALONE, never from the table caption. A caption can
# legitimately reference both macro AND micro scales for a table that only
# reports one of them (real corpus case: the Acta DP800 paper's caption
# "Comparison of the mechanical behavior at the macro and micro scale for
# both DP steel grades" sits over a table whose own YS/UTS columns are all
# unambiguously bulk/macroscale -- using the caption would have wrongly
# flagged genuine bulk data as ambiguous). Absence of a cue is
# 'unspecified' for both dimensions, never silently defaulted to
# 'bulk_or_macroscale'/'tensile'; a header carrying BOTH conflicting cues
# (e.g. mentions both "tensile" and "compressive") is also 'unspecified'
# rather than guessed. Checked in phase -> micro -> macro order for scope
# (a named phase implies non-bulk regardless of any accompanying
# macro/micro wording) since phase-level is the more specific signal.
_LOADING_COMPRESSIVE_RE = re.compile(r"compressive|compression", re.IGNORECASE)
_LOADING_TENSILE_RE = re.compile(r"\btensile\b|\btension\b", re.IGNORECASE)
_SCOPE_MICRO_RE = re.compile(r"micro[- ]?pillar|micropillar|\bpillars?\b|micro[- ]?scale|nano[- ]?indent", re.IGNORECASE)
_SCOPE_MACRO_RE = re.compile(r"macro[- ]?scale|\bmacro\b|\bbulk\b", re.IGNORECASE)
_PHASE_COMPONENT_RE = re.compile(r"\b(ferrite|martensite|austenite|bainite|pearlite|cementite)\b", re.IGNORECASE)


def classify_loading_mode(text: str) -> str:
    """'tensile' | 'compressive' | 'unspecified' -- never 'other' at this
    layer (a value only Stage 3's own duplicate copy needs to distinguish
    for grouping-key purposes is not introduced in Stage 2's candidate
    metadata, which stays purely descriptive)."""
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
    """'bulk_or_macroscale' | 'microscale_or_local' | 'phase_level' | 'unspecified'."""
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


PROCESSING_TERMS = [
    "anneal", "temper", "quench", "partition", "austenitize", "austenitise",
    "normalize", "normalise", "heat treatment", "heat-treatment", "soak",
    "intercritical", "q&p", "q-p", "q&t", "q-t", "austempering", "ausforming",
    "reheat", "hot-rolled", "cold-rolled", "hot rolling", "cold rolling",
    "holding time", "holding temperature",
]

TEST_CONDITION_TERMS = [
    "test temperature", "testing temperature", "strain rate", "strain-rate",
    "loading direction", "orientation", "shielding gas", "loading rate",
    "room temperature", "quasi-static", "specimen orientation",
]

PROPERTY_TERMS_PROSE = [
    "yield strength", "yield stress", "tensile strength", "ultimate tensile",
    "ultimate strength", "elongation", " uts ", " ys ", "proof stress",
]

COMPOSITION_TERMS_PROSE = [
    "composition", "wt.%", "wt%", "chemical composition", "alloying element",
    "carbon content", "manganese", "silicon", "chromium", "nickel",
    "molybdenum", "niobium", "titanium", "vanadium", "aluminum", "aluminium",
]

# General grade-label shape: 1-6 letters, optional "&", optional hyphen,
# then 1-4 digits — covers common conventions (DP780, DP1, TRIP780,
# HSLA450, Q&P980, X70) without hard-coding any specific grade name.
GRADE_LABEL_RE = re.compile(r"\b[A-Z]{1,6}(?:&[A-Z]{1,6})?-?\d{1,4}\b")

# Generic (not tied to any specific grade name) relational wording that a
# paper commonly uses when introducing a broader family/class alongside the
# specific grades studied, e.g. "two types of DP800 steel ... named DP1 and
# DP2" or "Two 7xxx-series aluminium alloys, Alloy A and Alloy B". This is
# used only as a corroborating diagnostic signal (material_identity_evidence)
# — it never by itself causes a material to be reclassified; see
# classify_material_identity's evidence-duplication test for the decisive
# (conservative, deterministic) signal.
_FAMILY_RELATION_WORDING_RE = re.compile(
    r"(?:types?|grades?|variants?|series)\s+of\s+[\w\-]+.{0,60}?(?:named|denoted|labell?ed)\s"
    r"|\b[\w\-]+[- ](?:series|type|grade|class)\s+[\w\-]+\s+(?:alloys?|steels?|grades?),",
    re.IGNORECASE,
)


def _has_family_relation_wording(text: str) -> bool:
    return bool(text) and bool(_FAMILY_RELATION_WORDING_RE.search(text))


REPLICATE_ROW_LABEL_RE = re.compile(r"^(specimen|sample|test|no\.?|trial)?\s*\d{1,2}$", re.IGNORECASE)
AGGREGATE_ROW_LABEL_TERMS = {"average", "avg", "mean", "sd", "sd *", "std", "std.", "std dev", "standard deviation"}

LITERATURE_REFERENCE_TERMS = [
    "ref.", "reference", "literature", "reported by", "standard", "specification",
    "astm", "iso ", "en 10025", "din ",
]
_BRACKETED_CITATION_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")


def _find_grade_labels(text: str) -> list[str]:
    if not text:
        return []
    return sorted(set(GRADE_LABEL_RE.findall(text)))


def _group_label_for_column(header_groups: list[dict], col_i: int) -> str | None:
    """Resolve which header_groups entry (if any) a 0-indexed column
    belongs to. Both table parsers (xml_to_json.py's _build_header_grid,
    springer_to_json.py's _jats_build_header_grid) document start_col/
    end_col as 1-indexed INCLUSIVE over the full row (including whatever
    is physically the first/label column) -- confirmed directly against
    the JATS parser's own col=1-start numbering. A 0-indexed column
    position col_i therefore corresponds to 1-indexed position col_i + 1."""
    pos = col_i + 1
    for g in header_groups or []:
        start, end = g.get("start_col"), g.get("end_col")
        if start is not None and end is not None and start <= pos <= end:
            return g.get("label")
    return None


def _match_property_alias_for_header(header: str) -> str | None:
    h = _normalize_for_match(header)
    for canonical, terms in PROPERTY_ALIAS_TERMS.items():
        if any(t in h for t in terms):
            return canonical
    if _is_bare_elongation_abbreviation(header or ""):
        return "elongation"
    if _is_bare_uts_symbol(header or ""):
        return "UTS"
    if _is_bare_ys_abbreviation(header or ""):
        return "YS"
    return None


def _match_property_alias_column_indices(headers: list[str]) -> dict[str, list[int]]:
    """Index-based, not header-string-based: a table can legitimately have
    two leaf columns with the identical header text (e.g. a 'GA' group and
    a 'WA' group each contributing their own 'AB'/'A' sub-columns) --
    matching by column position rather than by header string means every
    such column still gets its own independent candidate, never silently
    collapsed by a later duplicate-key overwrite."""
    out: dict[str, list[int]] = {"UTS": [], "YS": [], "elongation": []}
    for i, h in enumerate(headers):
        alias = _match_property_alias_for_header(h)
        if alias:
            out[alias].append(i)
    return out


def _composition_like_column_indices(headers: list[str]) -> list[int]:
    return [i for i, h in enumerate(headers) if h.strip().upper() in ELEMENT_SYMBOLS]


def _contains_any_term(text: str, terms: list[str]) -> bool:
    t = _normalize_for_match(text)
    return any(term in t for term in terms)


def _is_replicate_looking(row_label, all_rows: list) -> bool:
    if row_label is None:
        return False
    label = str(row_label).strip()
    if label.lower() in AGGREGATE_ROW_LABEL_TERMS:
        return False
    return bool(REPLICATE_ROW_LABEL_RE.match(label))


def _is_literature_reference_looking(caption: str, row_label) -> bool:
    text = f"{caption} {row_label or ''}"
    if _BRACKETED_CITATION_RE.search(text):
        return True
    return _contains_any_term(text, LITERATURE_REFERENCE_TERMS)


def _detect_paragraph_signals(text: str) -> dict:
    return {
        "material_grade_terms": _find_grade_labels(text),
        "composition_terms": [t for t in COMPOSITION_TERMS_PROSE if t in _normalize_for_match(text)],
        "processing_terms": [t for t in PROCESSING_TERMS if t in _normalize_for_match(text)],
        "property_terms": [t for t in PROPERTY_TERMS_PROSE if t in _normalize_for_match(text)],
        "test_condition_terms": [t for t in TEST_CONDITION_TERMS if t in _normalize_for_match(text)],
    }


def _paragraph_has_any_signal(signals: dict) -> bool:
    return any(signals.values())


# ══════════════════════════════════════════════════════════════════════════
# LAYER A — immutable deterministic candidate inventory. Contains NO
# LLM-derived fields anywhere. Built once, before any LLM call, never
# mutated afterward by any later layer.
# ══════════════════════════════════════════════════════════════════════════

def _build_table_candidates(paper: dict, id_index: dict) -> list[dict]:
    table_candidates = []
    for t_idx, table in enumerate(paper.get("tables", [])):
        table_id = f"T{t_idx}"
        headers = list(table.get("headers") or [])
        caption = table.get("caption", "") or ""
        rows_raw = table.get("rows", [])

        composition_like_column_indices = _composition_like_column_indices(headers)
        property_like_column_indices = _match_property_alias_column_indices(headers)
        # PATCH (row-oriented property table fallback): real corpus gap
        # (Springer/JATS 10.1007/s00170-026-17487-x, Table 2) -- some
        # tables report property names ("UTS (MPa)", "YS (MPa)") as ROW
        # labels rather than column headers, with columns instead carrying
        # condition/group labels ("AB"/"A" under "GA"/"WA" groups). The
        # column-header scan above then finds nothing at all. Invoked ONLY
        # when that scan found ZERO matches for every canonical property
        # in this WHOLE table -- a table that already has real UTS/YS/
        # elongation column headers never triggers this fallback, so a
        # table satisfying both orientations is never double-counted
        # (duplicate-protection by construction, not a special case).
        row_oriented_fallback_active = not any(property_like_column_indices.values())
        header_groups = table.get("header_groups") or []
        processing_term_columns = [h for h in headers if _contains_any_term(h, PROCESSING_TERMS)]
        test_condition_term_columns = [h for h in headers if _contains_any_term(h, TEST_CONDITION_TERMS)]
        candidate_material_labels_in_caption = _find_grade_labels(caption)
        all_row_provenance = table.get("row_provenance") or []

        id_index[table_id] = {
            "kind": "table",
            "location": {"type": "table", "table_index": t_idx, "row_index": None, "column": None},
            "raw_caption": caption,
            "raw_headers": list(headers),
            "accounting_candidate": False,  # structural/context anchor only — never part of the accounting closure
        }

        rows_out = []
        for r_idx, row in enumerate(rows_raw):
            row_id = f"{table_id}:R{r_idx}"
            row = list(row)
            row_prov = list(all_row_provenance[r_idx]) if r_idx < len(all_row_provenance) else []
            # Never positionally align a mismatched header/row pair — that
            # silently shifts values across columns (the exact failure mode
            # that mapped grain-size/martensite-fraction values onto
            # UTS/YS). Only build cell candidates when every header has a
            # corresponding cell one-to-one; otherwise this row contributes
            # no property/composition cell candidates, and the mismatch is
            # recorded on its id_index entry for audit.
            row_header_alignment = "resolved" if headers and len(headers) == len(row) else "table_header_row_mismatch"
            # Index-based only: candidate identity is tied to column index,
            # never to header text, so two columns that legitimately share
            # the same leaf header (e.g. a "GA" group's "AB" alongside a
            # "WA" group's own "AB") each still produce their own
            # independent candidate — nothing is silently overwritten.
            # `raw_cells` is retained only as a convenience/back-compat
            # summary (duplicate header text still collapses there), never
            # consulted for candidate generation below.
            raw_cells = dict(zip(headers, row)) if row_header_alignment == "resolved" else {}
            row_label = row[0] if row else None
            row_candidate_material_labels = _find_grade_labels(" ".join(str(c) for c in row))

            id_index[row_id] = {
                "kind": "table_row",
                "location": {"type": "table", "table_index": t_idx, "row_index": r_idx, "column": None},
                "raw_row": list(row),
                "raw_headers": list(headers),
                "raw_caption": caption,
                "row_label": row_label,
                "row_header_alignment": row_header_alignment,
                "candidate_material_labels": row_candidate_material_labels,
                "accounting_candidate": False,  # structural/context anchor only — never part of the accounting closure
            }

            cell_candidates = []
            if row_header_alignment == "resolved":
                for canonical, matched_indices in property_like_column_indices.items():
                    # PATCH: multiple distinct columns can legitimately map to
                    # the same canonical alias (e.g. "TS (MPa)" and "Rm" both
                    # -> UTS, or two same-named columns from different
                    # groups). Disambiguate with a numeric suffix so
                    # candidate IDs never collide, while canonical_alias
                    # stays the pure alias on every one of them.
                    alias_token = "ELONGATION" if canonical == "elongation" else canonical.upper()
                    present_indices = [i for i in matched_indices if i < len(row)]
                    for idx, col_i in enumerate(present_indices):
                        cid = f"{row_id}:C_{alias_token}" if idx == 0 else f"{row_id}:C_{alias_token}_{idx + 1}"
                        raw_value = row[col_i]
                        source_col = headers[col_i]
                        cell_prov = row_prov[col_i] if col_i < len(row_prov) else None
                        # Additive metadata only, never a separate numeric
                        # field: which elongation sub-type this column's
                        # wording indicates -- None for non-elongation
                        # properties (UTS/YS), never inferred from anything
                        # but the header text itself.
                        elongation_type = classify_elongation_type(source_col) if canonical == "elongation" else None
                        # Additive metadata only, from the column header
                        # text alone (never the caption -- see comment
                        # above classify_property_scope): which
                        # scope/loading-mode this column's wording
                        # indicates, only meaningful for UTS/YS (elongation
                        # has its own orthogonal sub-type dimension above).
                        if canonical in ("UTS", "YS"):
                            property_scope = classify_property_scope(source_col)
                            loading_mode = classify_loading_mode(source_col)
                        else:
                            property_scope = None
                            loading_mode = None
                        cell_candidates.append({"candidate_id": cid, "canonical_alias": canonical,
                                                 "source_column": source_col, "source_column_index": col_i,
                                                 "raw_value": raw_value, "row_id": row_id,
                                                 "elongation_type": elongation_type,
                                                 "property_scope": property_scope, "loading_mode": loading_mode})
                        id_index[cid] = {
                            "kind": "table_cell",
                            "location": {"type": "table", "table_index": t_idx, "row_index": r_idx, "column": source_col},
                            "canonical_alias": canonical, "source_column": source_col, "source_column_index": col_i,
                            "raw_value": raw_value, "cell_provenance": cell_prov, "elongation_type": elongation_type,
                            "property_scope": property_scope, "loading_mode": loading_mode,
                            "row_id": row_id, "raw_headers": list(headers), "raw_caption": caption,
                            "accounting_candidate": True,
                        }

                # Row-oriented fallback: this row's OWN label (first cell)
                # is checked against the identical, unmodified alias
                # matcher used for column headers above -- no new alias
                # system. A match means this whole row reports one
                # canonical property across its remaining (data) columns;
                # each such column becomes its own candidate, with
                # column-level group/condition ownership resolved from the
                # table's own already-parsed header_groups metadata.
                if row_oriented_fallback_active and row_label:
                    row_property_alias = _match_property_alias_for_header(str(row_label))
                    if row_property_alias:
                        row_label_text = str(row_label).strip()
                        row_alias_token = "ELONGATION" if row_property_alias == "elongation" else row_property_alias.upper()
                        # Same additive-metadata pattern as the column-oriented
                        # path above, derived from the ROW's own label text
                        # (which carries the property/unit signal) rather than
                        # the column header (which carries only the condition).
                        row_elongation_type = classify_elongation_type(row_label_text) if row_property_alias == "elongation" else None
                        if row_property_alias in ("UTS", "YS"):
                            row_property_scope = classify_property_scope(row_label_text)
                            row_loading_mode = classify_loading_mode(row_label_text)
                        else:
                            row_property_scope = None
                            row_loading_mode = None
                        for col_i in range(1, len(row)):
                            if col_i >= len(headers):
                                continue
                            cid = f"{row_id}:C_{row_alias_token}_col{col_i}"
                            raw_value = row[col_i]
                            # The REAL column header (e.g. "AB") -- required
                            # unchanged as both source_column and
                            # location["column"] so Stage 3's existing,
                            # unmodified resolve_column_index/read_location_
                            # value/reverify_property_item continue to
                            # correctly relocate and re-verify this exact
                            # cell (they match against the table's real
                            # headers, never against a synthesized string).
                            source_col = headers[col_i]
                            cell_prov = row_prov[col_i] if col_i < len(row_prov) else None
                            column_group_label = _group_label_for_column(header_groups, col_i)
                            cell_candidates.append({"candidate_id": cid, "canonical_alias": row_property_alias,
                                                     "source_column": source_col, "source_column_index": col_i,
                                                     "raw_value": raw_value, "row_id": row_id,
                                                     "elongation_type": row_elongation_type,
                                                     "property_scope": row_property_scope, "loading_mode": row_loading_mode,
                                                     "column_group_label": column_group_label,
                                                     "unit_source_text": row_label_text})
                            id_index[cid] = {
                                "kind": "table_cell",
                                "location": {"type": "table", "table_index": t_idx, "row_index": r_idx, "column": source_col},
                                "canonical_alias": row_property_alias, "source_column": source_col, "source_column_index": col_i,
                                "raw_value": raw_value, "cell_provenance": cell_prov, "elongation_type": row_elongation_type,
                                "property_scope": row_property_scope, "loading_mode": row_loading_mode,
                                "column_group_label": column_group_label, "unit_source_text": row_label_text,
                                "row_id": row_id, "raw_headers": list(headers), "raw_caption": caption,
                                "accounting_candidate": True,
                            }

                _seen_elements: dict = {}
                for col_i in composition_like_column_indices:
                    if col_i >= len(row):
                        continue
                    element = headers[col_i]
                    symbol = element.strip().upper()
                    n = _seen_elements.get(symbol, 0) + 1
                    _seen_elements[symbol] = n
                    cid = f"{row_id}:C_{symbol}" if n == 1 else f"{row_id}:C_{symbol}_{n}"
                    raw_value = row[col_i]
                    cell_prov = row_prov[col_i] if col_i < len(row_prov) else None
                    cell_candidates.append({"candidate_id": cid, "canonical_alias": symbol.capitalize(),
                                             "source_column": element, "source_column_index": col_i,
                                             "raw_value": raw_value, "row_id": row_id})
                    id_index[cid] = {
                        "kind": "table_cell",
                        "location": {"type": "table", "table_index": t_idx, "row_index": r_idx, "column": element},
                        "canonical_alias": symbol.capitalize(), "source_column": element, "source_column_index": col_i,
                        "raw_value": raw_value, "cell_provenance": cell_prov,
                        "row_id": row_id, "raw_headers": list(headers), "raw_caption": caption,
                        "is_composition_cell": True, "accounting_candidate": True,
                    }

            rows_out.append({
                "row_id": row_id, "row_index": r_idx, "row_label": row_label,
                "raw_cells": raw_cells,
                "signal_flags": {
                    "candidate_material_labels": row_candidate_material_labels,
                    "is_replicate_looking": _is_replicate_looking(row_label, rows_raw),
                    "is_literature_reference_looking": _is_literature_reference_looking(caption, row_label),
                },
                "cell_candidates": cell_candidates,
            })

        table_candidates.append({
            "table_id": table_id, "table_index": t_idx, "caption": caption, "headers": headers,
            "composition_like_column_indices": composition_like_column_indices,
            "property_like_column_indices": property_like_column_indices,
            "processing_term_columns": processing_term_columns,
            "test_condition_term_columns": test_condition_term_columns,
            "candidate_material_labels_in_caption": candidate_material_labels_in_caption,
            "rows": rows_out,
        })
    return table_candidates


def _build_section_candidates(paper: dict, id_index: dict) -> list[dict]:
    section_candidates = []
    for s_idx, section in enumerate(paper.get("sections", [])):
        section_id = f"S{s_idx}"
        heading = section.get("heading", "") or ""
        paragraphs_out = []
        for p_idx, para in enumerate(section.get("paragraphs", [])):
            signals = _detect_paragraph_signals(para)
            if not _paragraph_has_any_signal(signals):
                continue
            paragraph_id = f"{section_id}:P{p_idx}"
            id_index[paragraph_id] = {
                "kind": "section_paragraph",
                "location": {"type": "section", "heading": heading, "paragraph_index": p_idx},
                "raw_text": para, "heading": heading, "signals": signals,
                "accounting_candidate": True,
            }
            paragraphs_out.append({"paragraph_id": paragraph_id, "paragraph_index": p_idx,
                                    "raw_text": para, "signal_flags": signals})
        if paragraphs_out:
            section_candidates.append({"section_id": section_id, "heading": heading, "paragraphs": paragraphs_out})
    return section_candidates


def build_candidate_inventory(paper: dict, doi: str | None = None) -> dict:
    """The single Layer A entry point. Returns an inventory dict that must
    never be mutated after construction — every later layer only reads
    from it (directly, or via resolve_candidate_id)."""
    id_index: dict = {}
    table_candidates = _build_table_candidates(paper, id_index)
    section_candidates = _build_section_candidates(paper, id_index)
    return {
        "doi": doi or paper.get("doi", ""),
        "table_candidates": table_candidates,
        "section_candidates": section_candidates,
        "id_index": id_index,
    }


def resolve_candidate_id(inventory: dict, candidate_id: str) -> dict | None:
    """The ONE authoritative resolver. An ID absent from the immutable
    inventory returns None — it can never produce a location. Returns a
    COPY of the location, never a reference into the inventory's own dict,
    so a caller cannot accidentally mutate Layer A through this."""
    record = inventory["id_index"].get(candidate_id)
    if record is None:
        return None
    return dict(record["location"])


def all_accounting_candidate_ids(inventory: dict) -> set:
    """The accounting denominator: every candidate ID explicitly marked
    accounting_candidate=True by Layer A. Structural/context anchors (bare
    table IDs, row-container IDs) are marked False and are never part of
    this closure — they remain fully valid as evidence references and
    rendering anchors, but "every candidate must end in exactly one bucket"
    never applies to them, and force_unresolved_after_recovery() can never
    manufacture an unresolved_evidence entry for one, because unaccounted
    (below) only ever draws from this set."""
    return {cid for cid, rec in inventory["id_index"].items() if rec.get("accounting_candidate")}


def _labels_for_accounting_candidate(inventory: dict, rec: dict, known_grade_label: str | None = None) -> list[str]:
    """The grade labels detected near this accounting candidate — via its
    own row (for a table cell) or its own paragraph (for prose).

    PATCH (column-based material-identity accounting gap): row-text scanning
    alone is blind to a material expressed via TABLE COLUMN rather than row
    text (e.g. a row-oriented property table whose material groups are
    header_groups spanning several columns, never named in the row itself).
    Additively, for a table_cell candidate:
      - column_group_label (already resolved at candidate-generation time,
        e.g. "GA"/"WA"/"VIM/VAR") is always included when present — this is
        a structural fact about the candidate, not a guess.
      - the literal source_column header is included ONLY when it exactly
        matches known_grade_label — i.e. only when the caller is asking
        "does this candidate belong to THIS already-discovered material",
        never as a standalone new-grade detector. This is what keeps
        ordinary short condition/property headers (AB, A, AF, RT, MPa, %,
        ...) from ever being mistaken for a material grade: they are only
        ever compared against grade labels Stage 2 already discovered, and
        only ever for the one specific material being tested.
    Existing row-text-derived labels are always included unchanged."""
    if rec["kind"] == "table_cell":
        row_rec = inventory["id_index"].get(rec["row_id"])
        labels = list(row_rec.get("candidate_material_labels", [])) if row_rec else []
        group_label = rec.get("column_group_label")
        if group_label:
            labels.append(group_label)
        if known_grade_label:
            source_column = rec.get("source_column") or ""
            if source_column and _normalize_for_match(source_column) == _normalize_for_match(known_grade_label):
                labels.append(source_column)
        return labels
    if rec["kind"] == "section_paragraph":
        return rec.get("signals", {}).get("material_grade_terms", [])
    return []


LOCAL_PARAGRAPH_WINDOW = 2


def _local_section_paragraph_indices(inventory: dict, heading: str, center_idx: int) -> set:
    """Paragraph indices, among those already present in the inventory
    (i.e. already signal-bearing), within +/-LOCAL_PARAGRAPH_WINDOW of
    center_idx in the SAME section — never crossing a heading boundary."""
    return {
        rec["location"]["paragraph_index"]
        for rec in inventory["id_index"].values()
        if rec["kind"] == "section_paragraph" and rec["location"]["heading"] == heading
        and abs(rec["location"]["paragraph_index"] - center_idx) <= LOCAL_PARAGRAPH_WINDOW
    }


def build_material_candidate_pool(inventory: dict, material: dict) -> dict:
    """PATCH (approved clarification 2): a material's grouping scope is not
    restricted to physically-neighboring candidates only. Returns two
    DISTINCT, deterministic sets:
      local_candidate_ids       — same table as one of the material's own
                                   identifying/composition candidates, or a
                                   nearby paragraph (+/-LOCAL_PARAGRAPH_WINDOW)
                                   in the same section.
      broad_pool_candidate_ids  — every OTHER accounting candidate in the
                                   paper whose own local context is either
                                   (a) genuinely anonymous (no grade label
                                   detected anywhere near it — the "separate,
                                   unlabeled property table" case), or
                                   (b) labeled with THIS material's own
                                   grade_label, wherever it occurs.
    Pool membership is a SCOPE/VISIBILITY mechanism only — it determines
    what Call 2 gets to see, never what it may assign. A candidate whose
    local context is labeled with a DIFFERENT material's grade is excluded
    from this pool entirely, so it is never even offered to the wrong
    material's grouping call."""
    identifying_and_comp_ids = set(material.get("identifying_candidate_ids", [])) | set(material.get("composition_candidate_ids", []))

    local_table_indices = set()
    local_section_windows: dict[str, set] = {}
    for cid in identifying_and_comp_ids:
        rec = inventory["id_index"].get(cid)
        if rec is None:
            continue
        loc = rec["location"]
        if loc["type"] == "table":
            local_table_indices.add(loc["table_index"])
        elif loc["type"] == "section":
            local_section_windows.setdefault(loc["heading"], set()).update(
                _local_section_paragraph_indices(inventory, loc["heading"], loc["paragraph_index"]))

    local_ids, broad_pool_ids = set(), set()
    grade_label_norm = _normalize_for_match(material.get("grade_label", ""))

    for cid, rec in inventory["id_index"].items():
        if not rec.get("accounting_candidate"):
            continue
        loc = rec["location"]
        is_local = (
            (loc["type"] == "table" and loc["table_index"] in local_table_indices)
            or (loc["type"] == "section" and loc["paragraph_index"] in local_section_windows.get(loc["heading"], set()))
        )
        if is_local:
            # PATCH (row-ownership fix): table-level locality alone is not
            # enough when a single table contains multiple materials as
            # explicitly labelled rows (real corpus case: Acta DP800 paper's
            # Table 3, DP1 and DP2 each their own row) — the broad-pool
            # branch below already excludes a candidate explicitly labelled
            # with a DIFFERENT material's grade; the same check was missing
            # here, letting a same-table-different-row candidate through
            # unconditionally. A row with NO detected label (blank/merged
            # continuation row, or a genuinely anonymous table) is left
            # untouched — absence of a label is never grounds for exclusion,
            # only a positively-detected CONFLICTING label is.
            candidate_labels = _labels_for_accounting_candidate(inventory, rec, known_grade_label=material.get("grade_label"))
            if candidate_labels and grade_label_norm and not any(
                    _normalize_for_match(l) == grade_label_norm for l in candidate_labels):
                continue  # same table, but this row is explicitly labelled with a DIFFERENT material's grade
            local_ids.add(cid)
            continue

        candidate_labels = _labels_for_accounting_candidate(inventory, rec, known_grade_label=material.get("grade_label"))
        if not candidate_labels:
            broad_pool_ids.add(cid)  # anonymous — no detected label anywhere near it
        elif grade_label_norm and any(_normalize_for_match(l) == grade_label_norm for l in candidate_labels):
            broad_pool_ids.add(cid)  # this material's own label, elsewhere in the paper
        # else: labeled with a DIFFERENT material's grade -> excluded from this pool

    return {"local_candidate_ids": sorted(local_ids), "broad_pool_candidate_ids": sorted(broad_pool_ids)}


# ══════════════════════════════════════════════════════════════════════════
# LAYER B — semantic classification, entirely separate from the inventory.
# Every candidate ID an LLM (or, in this round, a mock/injected function)
# returns is validated against the immutable inventory before being trusted.
# ══════════════════════════════════════════════════════════════════════════

VALID_PROCESSING_STATE_TYPE = {"explicit_processing_condition", "baseline_or_as_received_state", "not_established"}
VALID_LINKAGE = {"explicitly_linked", "convincingly_linked", "unresolved_linkage"}
VALID_MODALITY = {"table", "text", "figure_only"}


def normalize_enum(value, valid_set: set, default: str) -> str:
    if isinstance(value, str):
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        if v in valid_set:
            return v
    return default


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


def _validate_ids(inventory: dict, raw_ids) -> tuple[list[str], list]:
    """Splits raw_ids into (valid, invalid). An invalid entry is never
    resolved to a location and is returned for explicit logging."""
    valid, invalid = [], []
    for cid in (raw_ids or []):
        if isinstance(cid, str) and resolve_candidate_id(inventory, cid) is not None:
            valid.append(cid)
        else:
            invalid.append(cid)
    return valid, invalid


def normalize_material_discovery_response(raw, inventory: dict) -> dict:
    materials, invalid_refs = [], []
    for m in (raw.get("materials") or []) if isinstance(raw, dict) else []:
        if not isinstance(m, dict):
            continue
        grade_label = m.get("grade_label") if isinstance(m.get("grade_label"), str) else ""
        identifying_ids, inv1 = _validate_ids(inventory, m.get("identifying_candidate_ids"))
        composition_ids, inv2 = _validate_ids(inventory, m.get("composition_candidate_ids"))
        invalid_refs += inv1 + inv2
        materials.append({"grade_label": grade_label, "identifying_candidate_ids": identifying_ids,
                           "composition_candidate_ids": composition_ids})
    return {"materials": materials, "invalid_candidate_refs": invalid_refs,
            "notes": raw.get("notes", "") if isinstance(raw, dict) else ""}


def normalize_grouping_response(raw, inventory: dict) -> dict:
    material_id = raw.get("material_id") if isinstance(raw, dict) and isinstance(raw.get("material_id"), str) else "M?"
    groups, invalid_refs = [], []
    for g in (raw.get("evidence_groups") or []) if isinstance(raw, dict) else []:
        if not isinstance(g, dict):
            continue
        proc_ids, inv1 = _validate_ids(inventory, g.get("processing_candidate_ids"))
        prop_ids, inv2 = _validate_ids(inventory, g.get("property_candidate_ids"))
        tc_ids, inv3 = _validate_ids(inventory, g.get("test_condition_candidate_ids"))
        invalid_refs += inv1 + inv2 + inv3
        groups.append({
            "group_id": g.get("group_id") if isinstance(g.get("group_id"), str) else "G?",
            "processing_candidate_ids": proc_ids, "property_candidate_ids": prop_ids,
            "test_condition_candidate_ids": tc_ids,
            "processing_state_type": normalize_enum(g.get("processing_state_type"), VALID_PROCESSING_STATE_TYPE, "not_established"),
            "linkage_status": normalize_enum(g.get("linkage_status"), VALID_LINKAGE, "unresolved_linkage"),
            "linkage_basis": g.get("linkage_basis") if isinstance(g.get("linkage_basis"), str) else "",
            "evidence_modality": normalize_enum(g.get("evidence_modality"), VALID_MODALITY, "text"),
        })

    known_non_target_ids, inv4 = _validate_ids(inventory, raw.get("known_non_target_candidate_ids") if isinstance(raw, dict) else None)
    unresolved_ids, inv5 = _validate_ids(inventory, raw.get("unresolved_candidate_ids") if isinstance(raw, dict) else None)
    invalid_refs += inv4 + inv5

    raw_reason_map = (raw.get("known_non_target_reason_map") if isinstance(raw, dict) else None) or {}
    reason_map = {k: v for k, v in raw_reason_map.items() if isinstance(k, str) and isinstance(v, str) and k in known_non_target_ids}

    return {
        "material_id": material_id,
        "evidence_groups": groups,
        "known_non_target_candidate_ids": known_non_target_ids,
        "known_non_target_reason_map": reason_map,
        "unresolved_candidate_ids": unresolved_ids,
        "invalid_candidate_refs": invalid_refs,
        "notes": raw.get("notes", "") if isinstance(raw, dict) else "",
    }


def _normalize_evidence_group(g, inventory: dict):
    if not isinstance(g, dict):
        return None, []
    proc_ids, inv1 = _validate_ids(inventory, g.get("processing_candidate_ids"))
    prop_ids, inv2 = _validate_ids(inventory, g.get("property_candidate_ids"))
    tc_ids, inv3 = _validate_ids(inventory, g.get("test_condition_candidate_ids"))
    group = {
        "group_id": g.get("group_id") if isinstance(g.get("group_id"), str) else "G?",
        "processing_candidate_ids": proc_ids, "property_candidate_ids": prop_ids,
        "test_condition_candidate_ids": tc_ids,
        "processing_state_type": normalize_enum(g.get("processing_state_type"), VALID_PROCESSING_STATE_TYPE, "not_established"),
        "linkage_status": normalize_enum(g.get("linkage_status"), VALID_LINKAGE, "unresolved_linkage"),
        "linkage_basis": g.get("linkage_basis") if isinstance(g.get("linkage_basis"), str) else "",
        "evidence_modality": normalize_enum(g.get("evidence_modality"), VALID_MODALITY, "text"),
    }
    return group, inv1 + inv2 + inv3


def normalize_recovery_response(raw, inventory: dict) -> dict:
    """The recovery schema differs from the per-material grouping schema:
    each recovered candidate may attach to an EXISTING material (by ID) or
    form a brand-new one Call 1 missed entirely — both are valid recovery
    outcomes per the approved recovery prompt design."""
    assignments, invalid_refs = [], []
    for a in (raw.get("recovered_assignments") or []) if isinstance(raw, dict) else []:
        if not isinstance(a, dict) or a.get("target") not in ("existing_material", "new_material"):
            continue
        groups = []
        for g in (a.get("evidence_groups") or []):
            group, inv = _normalize_evidence_group(g, inventory)
            if group is not None:
                groups.append(group)
            invalid_refs += inv

        if a.get("target") == "existing_material":
            material_id = a.get("material_id")
            if not isinstance(material_id, str):
                continue
            assignments.append({"target": "existing_material", "material_id": material_id, "evidence_groups": groups})
        else:
            grade_label = a.get("grade_label") if isinstance(a.get("grade_label"), str) else ""
            id_ids, inv4 = _validate_ids(inventory, a.get("identifying_candidate_ids"))
            comp_ids, inv5 = _validate_ids(inventory, a.get("composition_candidate_ids"))
            invalid_refs += inv4 + inv5
            assignments.append({"target": "new_material", "grade_label": grade_label,
                                 "identifying_candidate_ids": id_ids, "composition_candidate_ids": comp_ids,
                                 "evidence_groups": groups})

    known_non_target_ids, inv6 = _validate_ids(inventory, raw.get("known_non_target_candidate_ids") if isinstance(raw, dict) else None)
    unresolved_ids, inv7 = _validate_ids(inventory, raw.get("unresolved_candidate_ids") if isinstance(raw, dict) else None)
    invalid_refs += inv6 + inv7
    raw_reason_map = (raw.get("known_non_target_reason_map") if isinstance(raw, dict) else None) or {}
    reason_map = {k: v for k, v in raw_reason_map.items() if isinstance(k, str) and isinstance(v, str) and k in known_non_target_ids}

    return {
        "recovered_assignments": assignments,
        "known_non_target_candidate_ids": known_non_target_ids,
        "known_non_target_reason_map": reason_map,
        "unresolved_candidate_ids": unresolved_ids,
        "invalid_candidate_refs": invalid_refs,
        "notes": raw.get("notes", "") if isinstance(raw, dict) else "",
    }


# ══════════════════════════════════════════════════════════════════════════
# LAYER C — deterministic accounting audit. unaccounted and unresolved are
# computed and reported as structurally distinct sets, never conflated.
# PATCH (approved clarification 3): unaccounted is further decomposed into
# prompt_cap_omitted_ids (never shown to the LLM at all) and
# llm_unaccounted_ids (shown, but absent from every response bucket).
# ══════════════════════════════════════════════════════════════════════════

def compute_accounting(inventory: dict, materials_with_ids: list[dict], group_results: list[dict], presented_ids=None) -> dict:
    assigned = set()
    for m in materials_with_ids:
        assigned |= set(m["identifying_candidate_ids"]) | set(m["composition_candidate_ids"])
    known_non_target, unresolved = set(), set()
    for gr in group_results:
        for g in gr["evidence_groups"]:
            assigned |= set(g["processing_candidate_ids"]) | set(g["property_candidate_ids"]) | set(g["test_condition_candidate_ids"])
        known_non_target |= set(gr["known_non_target_candidate_ids"])
        unresolved |= set(gr["unresolved_candidate_ids"])

    accounting_ids = all_accounting_candidate_ids(inventory)
    # Defensive intersection: even if a structural ID were somehow cited in
    # one of these lists, it never counts toward the accounting closure —
    # clarification 1's guarantee, enforced here regardless of LLM output.
    assigned &= accounting_ids
    known_non_target &= accounting_ids
    unresolved &= accounting_ids
    accounted = assigned | known_non_target | unresolved
    unaccounted = accounting_ids - accounted

    if presented_ids is None:
        presented_ids = accounting_ids  # unit-test convenience default: assume everything was shown
    presented_ids = set(presented_ids) & accounting_ids
    prompt_cap_omitted_ids = (accounting_ids - presented_ids) & unaccounted
    llm_unaccounted_ids = unaccounted - prompt_cap_omitted_ids

    return {
        "assigned": assigned, "known_non_target": known_non_target, "unresolved": unresolved,
        "unaccounted": unaccounted,
        "prompt_cap_omitted_ids": prompt_cap_omitted_ids,
        "llm_unaccounted_ids": llm_unaccounted_ids,
        "all_accounting_candidate_ids": accounting_ids,
    }


def force_unresolved_after_recovery(accounting: dict, reason: str = "no_llm_classification_after_recovery_pass") -> list[dict]:
    """Deterministic, Python-authored conversion of whatever is STILL
    unaccounted after the one bounded recovery pass. Tagged source=
    'python_forced' so it is always distinguishable from an LLM's own
    explicit unresolved judgement (source='llm_explicit'). Only ever draws
    from `unaccounted`, which is itself only ever a subset of
    all_accounting_candidate_ids — a structural ID can never appear here."""
    return [{"candidate_id": cid, "reason": reason, "source": "python_forced"} for cid in sorted(accounting["unaccounted"])]


def build_recovery_scope(inventory: dict, prompt_cap_omitted_ids, llm_unaccounted_ids) -> dict:
    """PATCH (approved clarification 3): recovery scope is an ORDERED list —
    prompt-cap omissions first, genuine LLM misses after — so that if the
    recovery pass itself has a size cap, cap-omitted candidates (which never
    got a fair look at all) are never bumped by a lower-priority candidate
    the LLM already had a chance to classify."""
    ordered = list(sorted(prompt_cap_omitted_ids)) + list(sorted(llm_unaccounted_ids))
    items = []
    for cid in ordered:
        rec = inventory["id_index"].get(cid)
        if rec is None:
            continue
        entry = {"candidate_id": cid, "kind": rec["kind"], "location": rec["location"]}
        if rec["kind"] == "table_cell":
            entry.update({"source_column": rec["source_column"], "raw_value": rec["raw_value"], "raw_caption": rec.get("raw_caption", "")})
        elif rec["kind"] == "section_paragraph":
            entry.update({"raw_text": rec["raw_text"], "heading": rec.get("heading", "")})
        items.append(entry)
    return {"unaccounted_items": items, "ordered_candidate_ids": ordered}


def compute_accounting_rates(inventory: dict, accounting_pre_recovery: dict, unresolved_final: list[dict]) -> dict:
    """The three metrics from the approved design, computed as fractions of
    all_accounting_candidate_ids. llm_unaccounted_candidate_rate and
    prompt_cap_omission_rate are measured BEFORE the recovery pass (the
    real diagnostic signal); final_unresolved_candidate_rate is measured on
    the finished output (LLM-explicit + recovery-explicit + Python-forced,
    combined — status distinguishable per-item via each entry's 'source')."""
    total = len(accounting_pre_recovery["all_accounting_candidate_ids"]) or 1
    final_unresolved_ids = {u["candidate_id"] for u in unresolved_final}
    return {
        "prompt_cap_omission_rate": len(accounting_pre_recovery["prompt_cap_omitted_ids"]) / total,
        "llm_unaccounted_candidate_rate": len(accounting_pre_recovery["llm_unaccounted_ids"]) / total,
        "final_unresolved_candidate_rate": len(final_unresolved_ids) / total,
    }


# ══════════════════════════════════════════════════════════════════════════
# Deterministic failure-degradation fallback (Layer A alone, no LLM)
# ══════════════════════════════════════════════════════════════════════════

def deterministic_material_fallback(inventory: dict) -> list[dict]:
    """Used when material discovery fails entirely (or returns nothing
    despite the inventory containing real grade-label signals): every
    distinct grade label found anywhere in the inventory becomes a
    provisional, unclassified material with empty evidence so far."""
    labels: set[str] = set()
    for rec in inventory["id_index"].values():
        if rec["kind"] == "table_row":
            labels |= set(_find_grade_labels(" ".join(str(c) for c in rec["raw_row"])))
        elif rec["kind"] == "table":
            labels |= set(_find_grade_labels(rec["raw_caption"]))
        elif rec["kind"] == "section_paragraph":
            labels |= set(rec.get("signals", {}).get("material_grade_terms", []))
    return [{"grade_label": lbl, "identifying_candidate_ids": [], "composition_candidate_ids": []} for lbl in sorted(labels)]


# ══════════════════════════════════════════════════════════════════════════
# Resolution — candidate IDs -> the EXACT existing Stage 2 v1 evidence
# interface Stage 3 already consumes. location_verified is unconditionally
# True here, not because of a re-check, but because resolution is
# structurally impossible for an ID that isn't a real inventory entry in
# the first place -- there is no path to a location any other way.
# ══════════════════════════════════════════════════════════════════════════

def _resolve_evidence_item(inventory: dict, candidate_id: str) -> dict | None:
    rec = inventory["id_index"].get(candidate_id)
    if rec is None:
        return None
    # PATCH (row-level quote fallback fix, post-review): a table_row
    # reference has neither raw_value (that's cell-level) nor raw_text
    # (that's paragraph-level) — previously fell through to "", an empty
    # quote. Now resolves to the same space-joined whole-row rendering
    # already used elsewhere in this project for row-level text (matching
    # Stage 3's read_location_value() convention for the same case).
    if "raw_value" in rec:
        quote = rec["raw_value"]
    elif "raw_text" in rec:
        quote = rec["raw_text"]
    elif rec["kind"] == "table_row":
        quote = " ".join(str(c) for c in rec.get("raw_row", []))
    else:
        quote = ""
    return {"location": dict(rec["location"]), "quote": str(quote) if quote is not None else "",
            "candidate_id": candidate_id, "location_verified": True}


def resolve_material_to_output(inventory: dict, material: dict, material_id: str, group_result: dict | None) -> dict:
    composition_evidence = []
    for cid in material["composition_candidate_ids"]:
        item = _resolve_evidence_item(inventory, cid)
        if item is not None:
            composition_evidence.append(item)

    evidence_groups = []
    for g in (group_result["evidence_groups"] if group_result else []):
        processing_evidence = [it for cid in g["processing_candidate_ids"] if (it := _resolve_evidence_item(inventory, cid)) is not None]
        test_condition_evidence = [it for cid in g["test_condition_candidate_ids"] if (it := _resolve_evidence_item(inventory, cid)) is not None]
        property_evidence = []
        for cid in g["property_candidate_ids"]:
            rec = inventory["id_index"].get(cid)
            if rec is None or rec["kind"] != "table_cell":
                continue
            property_evidence.append({
                "property": rec["canonical_alias"], "location": dict(rec["location"]),
                "quote": str(rec.get("raw_value", "")), "candidate_id": cid, "location_verified": True,
                "elongation_type": rec.get("elongation_type"),
                "property_scope": rec.get("property_scope"), "loading_mode": rec.get("loading_mode"),
                "unit_source_text": rec.get("unit_source_text"),
            })
        evidence_groups.append({
            "group_id": g["group_id"], "processing_state_type": g["processing_state_type"],
            "processing_evidence": processing_evidence, "test_condition_evidence": test_condition_evidence,
            "property_evidence": property_evidence, "linkage_status": g["linkage_status"],
            "linkage_basis": g["linkage_basis"], "evidence_modality": g["evidence_modality"],
        })

    return {
        "material_id": material_id,
        "grade_label": material["grade_label"],
        "composition_evidence": composition_evidence,
        "composition_confidence": "medium",
        "composition_verified": len(composition_evidence) > 0,
        "evidence_groups": evidence_groups,
    }


def _material_evidence_candidate_ids(material: dict, group_result: dict | None) -> set:
    """The full set of evidence-bearing candidate IDs attributed to one
    discovered material: its composition candidates plus every property/
    processing/test-condition candidate across all its evidence groups.
    Identifying-only candidates (grade-label mentions with no evidence
    value) are deliberately excluded — a material referenced only by name,
    never by measured evidence, is not "duplicated evidence", it is no
    evidence at all, and is left as concrete_material (conservative
    default) by classify_material_identity below."""
    ids = set(material.get("composition_candidate_ids") or [])
    for g in (group_result["evidence_groups"] if group_result else []):
        ids |= set(g.get("property_candidate_ids") or [])
        ids |= set(g.get("processing_candidate_ids") or [])
        ids |= set(g.get("test_condition_candidate_ids") or [])
    return ids


def classify_material_identity(materials_with_ids: list[dict], group_results_by_material_id: dict,
                                inventory: dict | None = None) -> dict:
    """Deterministic, conservative post-grouping check distinguishing a
    genuinely independent (concrete_material) discovered material from one
    that is really a broader parent/family/class label the paper also
    mentions alongside the concrete grades it studied (family_or_class) --
    e.g. a paper that studies "DP1" and "DP2" but also names their shared
    class "DP800" in prose; the material-discovery LLM call sometimes
    (correctly, per its own "preserve distinct grades" instruction, which
    has no concept of a parent/family label) proposes the family name as if
    it were a third independent material.

    Requires MULTIPLE, structural signals before reclassifying — never a
    single weak signal:
      - the candidate's OWN full evidence-candidate-ID set (composition +
        every property/processing/test-condition candidate across all its
        groups) must be non-empty (a material with literally no evidence at
        all is a separate, pre-existing situation, not this one), AND
      - every one of those candidate IDs must ALSO appear in the evidence
        of at least one OTHER discovered material (i.e. zero unique
        evidence of its own -- full subsumption, not mere overlap).

    This explicitly does NOT use composition_verified==False alone (a
    material can legitimately have no composition table but genuine unique
    property evidence -- that must remain concrete_material), does not use
    substring/prefix matching on grade labels, and never hardcodes any
    specific label. If uncertain -- any unique evidence at all, or no other
    material to be redundant against -- the conservative default is
    concrete_material.

    Returns {material_id: {"material_identity_type": "concrete_material" |
    "family_or_class", "material_identity_evidence": <short reason string>}}."""
    per_material_ids = {
        m["material_id"]: _material_evidence_candidate_ids(m, group_results_by_material_id.get(m["material_id"]))
        for m in materials_with_ids
    }

    result = {}
    for m in materials_with_ids:
        mid = m["material_id"]
        own_ids = per_material_ids[mid]
        others_union = set().union(*(ids for other_mid, ids in per_material_ids.items() if other_mid != mid)) \
            if len(per_material_ids) > 1 else set()

        has_unique_evidence = bool(own_ids - others_union)

        if own_ids and others_union and not has_unique_evidence:
            reason = "all_evidence_duplicated_from_other_materials"
            if inventory is not None:
                # corroborating-only signal: does any of this material's own
                # candidate context read like a family/parent relational
                # sentence? Never decisive by itself (see docstring), but
                # useful for a human auditing the diagnostics.
                context_texts = []
                for cid in m.get("identifying_candidate_ids") or []:
                    rec = inventory["id_index"].get(cid)
                    if rec:
                        context_texts.append(str(rec.get("raw_text") or rec.get("raw_row") or ""))
                if any(_has_family_relation_wording(t) for t in context_texts):
                    reason += "+family_relation_wording_confirmed"
            result[mid] = {
                "material_identity_type": "family_or_class",
                "material_identity_evidence": reason,
            }
        elif not own_ids:
            result[mid] = {
                "material_identity_type": "concrete_material",
                "material_identity_evidence": "no_evidence_ids_present",
            }
        else:
            result[mid] = {
                "material_identity_type": "concrete_material",
                "material_identity_evidence": "has_unique_evidence" if has_unique_evidence else "sole_material_nothing_to_compare_against",
            }
    return result


def _empty_diagnostics() -> dict:
    """Zeroed, always-present shape for stage2_v2_diagnostics — instrumentation
    only, added post-hoc, never consulted by any decision logic. See module
    docstring section on diagnostics for what each field means."""
    return {
        "n_materials_from_call1": 0,
        "n_materials_final": 0,
        "n_materials_reclassified_as_family_or_class": 0,
        "call1": {"presented_count": 0, "prompt_cap_omitted_count": 0, "call_status": "not_run",
                   "secondary_fallback_used": False},
        "per_material_grouping": [],
        "recovery": {"triggered": False, "candidates_sent": 0, "prompt_cap_omitted_count": 0,
                     "call_status": "not_triggered", "recovered_count": 0},
        "accounting_pre_recovery_ids": {"unaccounted": [], "prompt_cap_omitted_ids": [], "llm_unaccounted_ids": []},
        "invalid_candidate_refs_by_call": {"call1": [], "grouping": {}, "recovery": []},
    }


def build_stage2_v2_output(doi: str, inventory: dict, materials_with_ids: list[dict],
                            group_results_by_material_id: dict, unresolved_final: list[dict],
                            known_non_target_final: list[dict], *, recovery_used: bool,
                            material_discovery_status: str, final_accounting: dict | None = None,
                            accounting_rates: dict | None = None,
                            invalid_candidate_refs: list | None = None,
                            diagnostics: dict | None = None,
                            family_materials_with_ids: list[dict] | None = None,
                            identity_classification: dict | None = None) -> dict:
    candidate_materials = []
    for m in materials_with_ids:
        resolved = resolve_material_to_output(inventory, m, m["material_id"], group_results_by_material_id.get(m["material_id"]))
        classification = (identity_classification or {}).get(m["material_id"], {})
        resolved["material_identity_type"] = classification.get("material_identity_type", "concrete_material")
        resolved["material_identity_evidence"] = classification.get("material_identity_evidence", "")
        candidate_materials.append(resolved)

    # Family/class candidates are never sent downstream as independent
    # materials (Stage 3 only ever reads candidate_materials), but their
    # full evidence is preserved here for audit — nothing discarded.
    family_candidates = []
    for m in (family_materials_with_ids or []):
        resolved = resolve_material_to_output(inventory, m, m["material_id"], group_results_by_material_id.get(m["material_id"]))
        classification = (identity_classification or {}).get(m["material_id"], {})
        resolved["material_identity_type"] = classification.get("material_identity_type", "family_or_class")
        resolved["material_identity_evidence"] = classification.get("material_identity_evidence", "")
        family_candidates.append(resolved)

    unresolved_evidence = []
    for u in unresolved_final:
        rec = inventory["id_index"].get(u["candidate_id"])
        if rec is None:
            continue
        unresolved_evidence.append({"location": dict(rec["location"]), "candidate_id": u["candidate_id"],
                                     "location_verified": True, "note": u["reason"], "source": u["source"]})

    known_non_target_evidence = []
    for k in known_non_target_final:
        rec = inventory["id_index"].get(k["candidate_id"])
        if rec is None:
            continue
        known_non_target_evidence.append({"reason": k["reason"], "location": dict(rec["location"]),
                                           "candidate_id": k["candidate_id"], "location_verified": True, "note": ""})

    return {
        "doi": doi,
        "candidate_materials": candidate_materials,
        "stage2_v2_family_or_class_candidates": family_candidates,
        "known_non_target_evidence": known_non_target_evidence,
        "unresolved_evidence": unresolved_evidence,
        "stage2_v2_material_discovery_status": material_discovery_status,
        "stage2_v2_recovery_pass_used": recovery_used,
        "stage2_v2_final_accounting": {k: sorted(v) for k, v in final_accounting.items()} if final_accounting else None,
        "stage2_v2_accounting_rates": accounting_rates,
        "stage2_v2_invalid_candidate_refs": invalid_candidate_refs or [],
        "stage2_v2_diagnostics": diagnostics if diagnostics is not None else _empty_diagnostics(),
        "code_version": CODE_VERSION,
        "prompt_version": PROMPT_VERSION,
    }


# ══════════════════════════════════════════════════════════════════════════
# Compact candidate rendering for each of the three LLM calls. Deterministic,
# offline-testable — returns exactly what was rendered AND which accounting
# candidates were left out purely due to the size cap, so that omission can
# never be confused with a genuine LLM classification miss.
# ══════════════════════════════════════════════════════════════════════════

MATERIAL_DISCOVERY_CANDIDATE_CAP = 150
GROUPING_CANDIDATE_CAP = 80
RECOVERY_CANDIDATE_CAP = 30
_PARAGRAPH_RENDER_CHAR_CAP = 400


def _render_single_candidate_line(inventory: dict, candidate_id: str, tier_label: str) -> str:
    rec = inventory["id_index"][candidate_id]
    if rec["kind"] == "table_cell":
        # column_group_label is present only for row-oriented-fallback
        # candidates (None/absent for the ordinary column-oriented path,
        # which renders exactly as before this patch) -- surfaces the
        # resolved higher-level group (e.g. "GA"/"WA") alongside the
        # column's own real header, so the LLM sees the disambiguating
        # context it needs without source_column ever being anything
        # other than the exact raw header string.
        group_label = rec.get("column_group_label")
        col_display = f"{group_label} / {rec['source_column']}" if group_label else rec['source_column']
        return f"{candidate_id} | {tier_label} | table \"{rec.get('raw_caption', '')}\" | {col_display}={rec['raw_value']}"
    if rec["kind"] == "table_row":
        cells = ", ".join(f"{h}={v}" for h, v in zip(rec["raw_headers"], rec["raw_row"]))
        return f"{candidate_id} | {tier_label} | table \"{rec.get('raw_caption', '')}\" | {cells}"
    if rec["kind"] == "section_paragraph":
        text = rec["raw_text"]
        if len(text) > _PARAGRAPH_RENDER_CHAR_CAP:
            text = text[:_PARAGRAPH_RENDER_CHAR_CAP] + "…"
        return f"{candidate_id} | {tier_label} | section \"{rec['heading']}\" | \"{text}\""
    return f"{candidate_id} | {tier_label} | (unrenderable kind {rec['kind']})"


def render_material_discovery_candidates(inventory: dict, cap: int = MATERIAL_DISCOVERY_CANDIDATE_CAP) -> dict:
    """Call 1 scope: the union of ALL accounting candidates, label-bearing
    ones first (a free, deterministic priority ordering) so a truncation
    under the cap always drops the lowest-value entries first."""
    accounting_ids = all_accounting_candidate_ids(inventory)
    label_bearing, other = [], []
    for cid in sorted(accounting_ids):
        rec = inventory["id_index"][cid]
        (label_bearing if _labels_for_accounting_candidate(inventory, rec) else other).append(cid)
    ordered = label_bearing + other
    presented, omitted = ordered[:cap], ordered[cap:]
    lines = [_render_single_candidate_line(inventory, cid, "LABEL_BEARING" if cid in label_bearing else "OTHER_SIGNAL")
             for cid in presented]
    text = "\n".join(lines)
    if omitted:
        text += f"\n(...{len(omitted)} additional signal-bearing candidates not shown due to size limits)"
    return {"rendered_text": text, "presented_ids": presented, "prompt_cap_omitted_ids": omitted}


def render_grouping_candidates(inventory: dict, local_candidate_ids, broad_pool_candidate_ids,
                                cap: int = GROUPING_CANDIDATE_CAP) -> dict:
    """Local candidates are always prioritized over broad-pool ones under
    the cap. Broad-pool entries are rendered with an explicit tier label —
    visibility/context only, never linkage evidence in themselves."""
    local_set = set(local_candidate_ids)
    ordered = list(sorted(local_candidate_ids)) + list(sorted(broad_pool_candidate_ids))
    presented, omitted = ordered[:cap], ordered[cap:]
    lines = [_render_single_candidate_line(inventory, cid, "LOCAL" if cid in local_set else "BROAD_POOL (context only, not linkage evidence)")
             for cid in presented]
    text = "\n".join(lines)
    if omitted:
        text += f"\n(...{len(omitted)} additional candidates not shown due to size limits)"
    return {"rendered_text": text, "presented_ids": presented, "prompt_cap_omitted_ids": omitted}


def render_recovery_candidates(inventory: dict, ordered_candidate_ids: list, cap: int = RECOVERY_CANDIDATE_CAP) -> dict:
    presented, omitted = ordered_candidate_ids[:cap], ordered_candidate_ids[cap:]
    lines = [_render_single_candidate_line(inventory, cid, "RECOVERY") for cid in presented]
    text = "\n".join(lines)
    if omitted:
        text += f"\n(...{len(omitted)} additional unaccounted candidates not shown due to size limits — will remain unresolved this round)"
    return {"rendered_text": text, "presented_ids": presented, "prompt_cap_omitted_ids": omitted}


# ══════════════════════════════════════════════════════════════════════════
# Top-level orchestration. Takes INJECTABLE call functions so the full flow
# (including all failure-degradation paths and the rendering/pool/accounting
# wiring) is testable offline without ever touching Ollama. A real caller
# would pass functions that call the LLM with the given rendered text; this
# round's tests pass mocks/fakes only.
# ══════════════════════════════════════════════════════════════════════════

def _classify_llm_call_error(e: Exception) -> str:
    """Diagnostic-only: distinguishes a genuine call failure from the
    synthetic LLMCallError this module itself raises when a response parses
    to no JSON object. Never affects control flow — both cases are handled
    identically by the (unchanged) except blocks that call this."""
    return "malformed_json" if "malformed JSON" in str(e) else "llm_call_failed"


def process_paper_v2(paper: dict, doi: str, material_discovery_call_fn, grouping_call_fn, recovery_call_fn=None) -> dict:
    inventory = build_candidate_inventory(paper, doi=doi)
    accounting_ids = all_accounting_candidate_ids(inventory)

    if not accounting_ids:
        return build_stage2_v2_output(doi, inventory, [], {}, [], [], recovery_used=False,
                                       material_discovery_status="empty_inventory",
                                       diagnostics=_empty_diagnostics())

    invalid_refs: list = []
    presented_ids: set = set()
    invalid_refs_by_call: dict = {"call1": [], "grouping": {}, "recovery": []}

    # NOTE: caps are passed explicitly (not relying on the functions' own
    # default parameter values) so that a caller reconfiguring the module-
    # level cap constants at runtime is respected here — Python binds
    # default argument values once at function-definition time, so relying
    # on them would silently ignore any later reassignment of e.g.
    # MATERIAL_DISCOVERY_CANDIDATE_CAP.
    render1 = render_material_discovery_candidates(inventory, cap=MATERIAL_DISCOVERY_CANDIDATE_CAP)
    presented_ids |= set(render1["presented_ids"])
    call1_status = "ok"  # diagnostic only
    try:
        raw_text = material_discovery_call_fn(render1["rendered_text"])
        raw_json = parse_llm_response(raw_text)
        if raw_json is None:
            raise LLMCallError("malformed JSON from material discovery call")
        discovery = normalize_material_discovery_response(raw_json, inventory)
        materials = discovery["materials"]
        invalid_refs += discovery["invalid_candidate_refs"]
        invalid_refs_by_call["call1"] = list(discovery["invalid_candidate_refs"])
        material_discovery_status = "ok"
    except LLMCallError as e:
        materials = deterministic_material_fallback(inventory)
        material_discovery_status = f"llm_call_failed_deterministic_fallback: {e}"
        call1_status = _classify_llm_call_error(e)

    call1_secondary_fallback_used = False  # diagnostic only
    if not materials:
        fallback = deterministic_material_fallback(inventory)
        if fallback:
            materials = fallback
            material_discovery_status += "+deterministic_fallback_labels_found"
            call1_secondary_fallback_used = True

    materials_with_ids = []
    for i, m in enumerate(materials, 1):
        m = dict(m)
        m["material_id"] = f"M{i}"
        materials_with_ids.append(m)

    n_materials_from_call1 = len(materials_with_ids)  # diagnostic only — before recovery may add more

    per_material_diag: list = []  # diagnostic only
    group_results_by_material_id: dict = {}
    for m in materials_with_ids:
        pool = build_material_candidate_pool(inventory, m)
        render2 = render_grouping_candidates(inventory, pool["local_candidate_ids"], pool["broad_pool_candidate_ids"],
                                              cap=GROUPING_CANDIDATE_CAP)
        presented_ids |= set(render2["presented_ids"])
        material_call_status = "ok"  # diagnostic only
        try:
            raw_text = grouping_call_fn(m["material_id"], render2["rendered_text"])
            raw_json = parse_llm_response(raw_text)
            if raw_json is None:
                raise LLMCallError("malformed JSON from grouping call")
            result = normalize_grouping_response(raw_json, inventory)
            invalid_refs += result["invalid_candidate_refs"]
            invalid_refs_by_call["grouping"][m["material_id"]] = list(result["invalid_candidate_refs"])
            group_results_by_material_id[m["material_id"]] = result
        except LLMCallError as e:
            group_results_by_material_id[m["material_id"]] = {
                "material_id": m["material_id"], "evidence_groups": [],
                "known_non_target_candidate_ids": [], "known_non_target_reason_map": {},
                "unresolved_candidate_ids": [], "invalid_candidate_refs": [], "notes": "",
            }
            invalid_refs_by_call["grouping"][m["material_id"]] = []
            material_call_status = _classify_llm_call_error(e)
        per_material_diag.append({
            "material_id": m["material_id"],
            "presented_count": len(render2["presented_ids"]),
            "prompt_cap_omitted_count": len(render2["prompt_cap_omitted_ids"]),
            "call_status": material_call_status,
        })

    accounting_pre_recovery = compute_accounting(inventory, materials_with_ids,
                                                  list(group_results_by_material_id.values()), presented_ids=presented_ids)
    accounting_pre_recovery_ids = {  # diagnostic only — raw ID sets, not just rates
        "unaccounted": sorted(accounting_pre_recovery["unaccounted"]),
        "prompt_cap_omitted_ids": sorted(accounting_pre_recovery["prompt_cap_omitted_ids"]),
        "llm_unaccounted_ids": sorted(accounting_pre_recovery["llm_unaccounted_ids"]),
    }

    unresolved_final = [{"candidate_id": cid, "reason": "llm_explicit_unresolved", "source": "llm_explicit"}
                         for gr in group_results_by_material_id.values() for cid in gr["unresolved_candidate_ids"]]
    known_non_target_final = [{"candidate_id": cid, "reason": gr["known_non_target_reason_map"].get(cid, "other_non_target"), "source": "llm_explicit"}
                               for gr in group_results_by_material_id.values() for cid in gr["known_non_target_candidate_ids"]]

    recovery_diag = {"triggered": False, "candidates_sent": 0, "prompt_cap_omitted_count": 0,  # diagnostic only
                      "call_status": "not_triggered", "recovered_count": 0}

    accounting = accounting_pre_recovery
    recovery_used = False
    if accounting_pre_recovery["unaccounted"] and recovery_call_fn is not None:
        recovery_used = True
        recovery_diag["triggered"] = True
        recovery_scope = build_recovery_scope(inventory, accounting_pre_recovery["prompt_cap_omitted_ids"],
                                               accounting_pre_recovery["llm_unaccounted_ids"])
        render3 = render_recovery_candidates(inventory, recovery_scope["ordered_candidate_ids"], cap=RECOVERY_CANDIDATE_CAP)
        presented_ids |= set(render3["presented_ids"])
        recovery_diag["candidates_sent"] = len(render3["presented_ids"])
        recovery_diag["prompt_cap_omitted_count"] = len(render3["prompt_cap_omitted_ids"])
        try:
            raw_text = recovery_call_fn(render3["rendered_text"])
            raw_json = parse_llm_response(raw_text)
            if raw_json is None:
                raise LLMCallError("malformed JSON from recovery call")
            recovery_result = normalize_recovery_response(raw_json, inventory)
            invalid_refs += recovery_result["invalid_candidate_refs"]
            invalid_refs_by_call["recovery"] = list(recovery_result["invalid_candidate_refs"])

            for a in recovery_result["recovered_assignments"]:
                if a["target"] == "existing_material" and a["material_id"] in group_results_by_material_id:
                    existing = group_results_by_material_id[a["material_id"]]
                    existing["evidence_groups"] = existing["evidence_groups"] + a["evidence_groups"]
                elif a["target"] == "new_material":
                    new_id = f"M{len(materials_with_ids) + 1}"
                    new_material = {"grade_label": a["grade_label"], "identifying_candidate_ids": a["identifying_candidate_ids"],
                                     "composition_candidate_ids": a["composition_candidate_ids"], "material_id": new_id}
                    materials_with_ids.append(new_material)
                    group_results_by_material_id[new_id] = {
                        "material_id": new_id, "evidence_groups": a["evidence_groups"],
                        "known_non_target_candidate_ids": [], "known_non_target_reason_map": {},
                        "unresolved_candidate_ids": [], "invalid_candidate_refs": [], "notes": "",
                    }

            for cid in recovery_result["known_non_target_candidate_ids"]:
                known_non_target_final.append({"candidate_id": cid, "reason": recovery_result["known_non_target_reason_map"].get(cid, "other_non_target"), "source": "llm_explicit_recovery"})
            for cid in recovery_result["unresolved_candidate_ids"]:
                unresolved_final.append({"candidate_id": cid, "reason": "llm_explicit_unresolved_recovery", "source": "llm_explicit"})
            recovery_diag["call_status"] = "ok"
        except LLMCallError as e:
            recovery_diag["call_status"] = _classify_llm_call_error(e)
            # recovery failure: prior classifications are untouched, nothing merged, nothing lost

        accounting = compute_accounting(inventory, materials_with_ids, list(group_results_by_material_id.values()),
                                         presented_ids=presented_ids)
        recovery_diag["recovered_count"] = len(accounting_pre_recovery["unaccounted"]) - len(accounting["unaccounted"])

    forced = force_unresolved_after_recovery(accounting)
    unresolved_final = unresolved_final + forced

    accounting_rates = compute_accounting_rates(inventory, accounting_pre_recovery, unresolved_final)

    # Deterministic, conservative post-grouping check: a discovered
    # "material" whose entire evidence is duplicated from another material
    # already discovered (e.g. a paper's broader steel-family label
    # alongside the specific grades it studied) is reclassified as
    # family_or_class and excluded from candidate_materials, never merged
    # or deleted — see classify_material_identity's docstring.
    identity_classification = classify_material_identity(materials_with_ids, group_results_by_material_id, inventory)
    concrete_materials_with_ids = [m for m in materials_with_ids
                                    if identity_classification[m["material_id"]]["material_identity_type"] == "concrete_material"]
    family_materials_with_ids = [m for m in materials_with_ids
                                  if identity_classification[m["material_id"]]["material_identity_type"] == "family_or_class"]

    diagnostics = {
        "n_materials_from_call1": n_materials_from_call1,
        "n_materials_final": len(materials_with_ids),
        "n_materials_reclassified_as_family_or_class": len(family_materials_with_ids),
        "call1": {"presented_count": len(render1["presented_ids"]), "prompt_cap_omitted_count": len(render1["prompt_cap_omitted_ids"]),
                  "call_status": call1_status, "secondary_fallback_used": call1_secondary_fallback_used},
        "per_material_grouping": per_material_diag,
        "recovery": recovery_diag,
        "accounting_pre_recovery_ids": accounting_pre_recovery_ids,
        "invalid_candidate_refs_by_call": invalid_refs_by_call,
    }

    return build_stage2_v2_output(
        doi, inventory, concrete_materials_with_ids, group_results_by_material_id,
        unresolved_final, known_non_target_final,
        recovery_used=recovery_used, material_discovery_status=material_discovery_status,
        final_accounting=accounting, accounting_rates=accounting_rates, invalid_candidate_refs=invalid_refs,
        diagnostics=diagnostics,
        family_materials_with_ids=family_materials_with_ids,
        identity_classification=identity_classification,
    )


# ══════════════════════════════════════════════════════════════════════════
# LLM call scaffold — self-contained copy of Stage 3's now-corrected
# caller-side hard-timeout mechanism (daemon thread + Thread.join(timeout=
# ...), not the earlier broken ThreadPoolExecutor-as-context-manager
# pattern). Present so this file compiles end-to-end and is ready to be
# wired up in a future round; NOT invoked anywhere in this module or its
# test suite this round. No prompt content has been designed/reviewed yet.
# ══════════════════════════════════════════════════════════════════════════

class LLMCallError(Exception):
    pass


def _post_to_ollama(prompt: str):
    return requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": 32768},
    }, timeout=LLM_CALL_TIMEOUT_S)


def call_llm(prompt: str) -> str:
    """Caller-side hard wall-clock deadline via a daemon thread — same
    corrected mechanism as Stage 3's call_llm(). It bounds how long this
    pipeline waits, not the underlying Ollama-side generation itself."""
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


# ══════════════════════════════════════════════════════════════════════════
# Prompt assembly — the SYSTEM_PROMPT text below is the verbatim,
# previously-approved design (design-review round preceding implementation),
# with one small, necessary addition: the grouping prompt now explains the
# LOCAL/BROAD_POOL candidate tier labels that render_grouping_candidates()
# actually produces (a rendering format introduced by the later-approved
# broad-pool clarification, after the original prompt text was drafted).
# This is wiring already-approved text into callable form for the first
# time — no heuristic, schema, cap, or decision rule changes.
# ══════════════════════════════════════════════════════════════════════════

MATERIAL_DISCOVERY_SYSTEM_PROMPT = """\
You are identifying distinct MATERIALS (steel grades/alloys) discussed in a
steel materials science paper, for a deterministic extraction pipeline.

You are NOT reading numeric values, NOT deciding whether this paper is
relevant, and NOT grouping evidence into conditions — those are separate,
later steps. Your only job: how many distinct materials/grades does this
paper discuss, and which of the supplied candidate IDs identify or belong
to each one?

You are shown a list of CANDIDATE IDs, each with a short piece of local
context (a table row, or a paragraph snippet). These candidates were
already generated by deterministic code from the real paper — they are
location markers, not numbers you need to verify. Many are false positives
(the candidate list favors completeness over precision); that is expected.

CRITICAL RULE — CANDIDATE IDs ONLY: every ID you return must be copied
EXACTLY, character for character, from the candidate_id values shown below.
Never write your own table/row/column/paragraph reference. Never alter or
abbreviate a candidate_id. An ID that does not exactly match one you were
shown will be discarded.

CRITICAL RULE — GROUNDED LABELS ONLY: only report a material whose grade
label is visibly supported by the context shown for at least one candidate
you cite for it. Do not invent a label that isn't visible in what you were
shown.

CRITICAL RULE — PRESERVE DISTINCT GRADES: if the paper distinguishes
several related grades (for example labeled with different numbers or
different processing suffixes), treat them as SEPARATE materials. Do not
collapse them into one entry merely because they share a family name or
appear in the same table.

CRITICAL RULE — SCOPE: do not create evidence groups. Do not classify
composition vs. processing vs. property vs. test-condition beyond
identifying which candidates report this material's composition. Do not
extract or report any numeric value. Do not judge whether this paper is
CPP-relevant.

Return ONLY a valid JSON object with this structure — no extra text:
{
  "materials": [
    {
      "grade_label": "as it appears in the source",
      "identifying_candidate_ids": ["candidate ids that name or identify this material"],
      "composition_candidate_ids": ["candidate ids that report this material's composition, if visible"]
    }
  ],
  "notes": "one short sentence"
}

If you are not confident a label is real, do not include it — a missed
material can still be found later; do not guess.
"""

GROUPING_SYSTEM_PROMPT = """\
You are classifying evidence CANDIDATES for ONE material already identified
in a steel materials science paper, for a deterministic extraction
pipeline. You are given this material's own identifying candidates, plus
every other candidate relevant to it.

Each candidate line is tagged LOCAL or BROAD_POOL. LOCAL candidates are
physically near this material's own identifying evidence (same table, or a
nearby paragraph). BROAD_POOL candidates are shown for additional
recall/context — they may be anonymous evidence elsewhere in the paper that
names no material at all, or evidence elsewhere that happens to share this
material's own label. Being shown to you as BROAD_POOL is NOT itself
evidence that a candidate belongs to this material — apply the exact same
three-way agreement rule below to BROAD_POOL candidates as to LOCAL ones.

CRITICAL RULE — CANDIDATE IDs ONLY: every ID you return must be copied
EXACTLY from the candidate_id values shown to you. Never write your own
table/row/column/paragraph reference.

CRITICAL RULE — CLASSIFY HONESTLY, DO NOT FORCE: many candidates shown to
you are false positives from a deliberately over-inclusive detector. Your
job is NOT to use every candidate — it is to classify each one honestly.
Every candidate you are shown must end up in exactly one of: an evidence
group's candidate lists, known_non_target_candidate_ids (with a reason), or
unresolved_candidate_ids. Do not omit any candidate from your response. If
you are not confident, mark it unresolved rather than guessing — a missed
candidate can still be recovered later, but a wrongly grouped one causes
real harm downstream.

CRITICAL RULE — GROUPING REQUIRES THREE INDEPENDENT AGREEMENTS: candidates
may only join the SAME evidence_group if you can independently support ALL
THREE of: (1) the same material state (already given — you are working
within one material), (2) the same PROCESSING state, and (3) the same TEST
condition. Being about the same material is never enough by itself.

  - Do NOT combine a standard/as-received specimen state with a
    pre-strained, backed, or otherwise separately-processed/tested specimen
    state — even if both concern the same material, values from a table
    describing routine testing must not be combined with values from a
    table describing a different specimen history.
  - Do NOT treat a sweep of test temperatures or strain rates as separate
    PROCESSING conditions. A material tested at several temperatures under
    ONE processing route is one processing condition with several test
    conditions, not several processing conditions.
  - Do NOT turn replicate specimens (repeated measurements of the same
    condition) into separate processing conditions — group them as
    multiple property_candidate_ids within ONE evidence_group.
  - Do NOT merge different grades into one group merely because they
    belong to the same steel family. You are only working within one
    already-identified material in this call; do not widen scope to a
    sibling grade's evidence.

CRITICAL RULE — PROCESSING vs. TEST CONDITION: a step describing
manufacturing/heat-treatment history is processing evidence. A parameter
describing how a specimen was TESTED (temperature, strain rate, loading
direction) is test-condition evidence, even if it also varies
specimen-to-specimen.

CRITICAL RULE ON UNCERTAINTY: if you cannot positively establish all three
required agreements for a candidate, leave it in unresolved_candidate_ids
rather than forcing an assignment. Correctness matters more than
completeness — that is what the recovery pass and downstream verification
are for.

Return ONLY a valid JSON object with this structure — no extra text:
{
  "material_id": "the material_id you were given",
  "evidence_groups": [
    {
      "group_id": "G1",
      "processing_candidate_ids": [],
      "property_candidate_ids": [],
      "test_condition_candidate_ids": [],
      "processing_state_type": "one value from the list below",
      "linkage_status": "one value from the list below",
      "linkage_basis": "one short sentence explaining the link",
      "evidence_modality": "one value from the list below"
    }
  ],
  "known_non_target_candidate_ids": [],
  "known_non_target_reason_map": {},
  "unresolved_candidate_ids": [],
  "notes": "one short sentence"
}

Valid processing_state_type values: explicit_processing_condition,
baseline_or_as_received_state, not_established.
Valid linkage_status values: explicitly_linked, convincingly_linked,
unresolved_linkage.
Valid evidence_modality values: table, text, figure_only.

EVERY candidate_id shown to you below must appear in exactly one of: some
evidence_group's candidate lists, known_non_target_candidate_ids, or
unresolved_candidate_ids.
"""

RECOVERY_SYSTEM_PROMPT = """\
You are examining candidates that were NOT classified by an earlier pass
over this paper's evidence. This is a SECOND, TARGETED look at a small set
of specific candidates — not a re-read of the paper. You are given the
unaccounted candidates themselves with their local source text, and a
short summary of materials/evidence groups already established.

CRITICAL RULE — CANDIDATE IDs ONLY: every ID you return must be copied
exactly from what you were shown.

CRITICAL RULE — SCOPE: for each candidate shown to you, decide: does it
belong to an EXISTING material/group (reference it by ID), does it require
a NEW material or evidence group, is it known non-target, or is it
unresolved. Do not revisit or reclassify anything not shown to you here —
those candidates were already explicitly examined in an earlier pass and
are not part of this one.

CRITICAL RULE — THIS IS THE ONLY RECOVERY ATTEMPT: if you cannot confidently
classify a candidate now, mark it unresolved. It will not be re-examined
again after this.

Return ONLY a valid JSON object with this structure — no extra text:
{
  "recovered_assignments": [
    {
      "target": "existing_material",
      "material_id": "an existing material_id you were shown",
      "evidence_groups": []
    },
    {
      "target": "new_material",
      "grade_label": "as it appears in the source",
      "identifying_candidate_ids": [],
      "composition_candidate_ids": [],
      "evidence_groups": []
    }
  ],
  "known_non_target_candidate_ids": [],
  "known_non_target_reason_map": {},
  "unresolved_candidate_ids": [],
  "notes": "one short sentence"
}

An evidence_groups entry uses the same shape as the grouping step:
{"group_id": "G1", "processing_candidate_ids": [], "property_candidate_ids": [],
 "test_condition_candidate_ids": [], "processing_state_type": "...",
 "linkage_status": "...", "linkage_basis": "...", "evidence_modality": "..."}
(same valid-value lists as before). Every candidate_id shown to you below
must appear in exactly one of: a recovered_assignments entry's IDs,
known_non_target_candidate_ids, or unresolved_candidate_ids.
"""


def build_material_discovery_prompt(rendered_text: str) -> str:
    return (f"{MATERIAL_DISCOVERY_SYSTEM_PROMPT}\n\nCANDIDATES:\n{rendered_text}\n\n"
            f"Respond with ONLY the JSON object now. No other text.\n")


def build_grouping_prompt(material_id: str, rendered_text: str) -> str:
    return (f"{GROUPING_SYSTEM_PROMPT}\n\nYou are working on material_id \"{material_id}\".\n\n"
            f"CANDIDATES:\n{rendered_text}\n\nRespond with ONLY the JSON object now. No other text.\n")


def build_recovery_prompt(rendered_text: str, materials_summary: str = "") -> str:
    summary = materials_summary or "(no materials established yet)"
    return (f"{RECOVERY_SYSTEM_PROMPT}\n\nALREADY-ESTABLISHED MATERIALS/GROUPS:\n{summary}\n\n"
            f"UNACCOUNTED CANDIDATES:\n{rendered_text}\n\nRespond with ONLY the JSON object now. No other text.\n")


def doi_to_processed_filename(doi: str) -> str:
    return doi.replace(".", "-").replace("/", "_") + ".json"
