"""
build_observation_dataset.py — Stage 5 (dataset construction): flattens Stage 3's
hierarchical output (paper -> material -> processing_condition -> observation)
into the one-row-per-valid-observation modelling table that
literature_pipeline.modelling.dataset_contract validates and
literature_pipeline.modelling.modelling_lib consumes.

── Scope ─────────────────────────────────────────────────────────────────────
Deterministic only. No LLM calls, no modelling, no imputation/scaling/one-hot
encoding, no GroupKFold, no quality-tier filtering — all of that is
modelling_lib.py's/§3.9's job, not this file's. This module only transforms
already-produced Stage 3 JSON into a flat table plus audit/QC/duplicate
reports; it never re-derives a value from the original processed-paper JSON
and never touches Stage 1/2/3, dataset_contract.py, or modelling_lib.py.

── Row unit (approved design) ─────────────────────────────────────────────────
Only Stage 3 records with observation_status == "valid_observation" become
modelling rows. orphan_property_evidence, rejected_candidate, and failures
are never modelling rows — they are preserved, with reasons, in the excluded-
records output. A valid_observation whose only property value(s) are None
because of an unresolved replicate cluster (Stage 3's own convention: a
property with more than one raw replicate item and no single positively-
compatible value) is ALSO excluded from the modelling CSV if every one of its
present targets is unresolved this way — but it is preserved in the excluded
output with the explicit reason "replicate_target_unresolved", never silently
dropped. No averaging/median/first-value selection happens anywhere in this
file.

── Composition ────────────────────────────────────────────────────────────────
Stage 3 stores composition once per material. This module propagates the same
values onto every one of that material's accepted observation rows (composition
is a reference, not a per-observation fact, per Stage 3's own design) — the
numeric element value and its qualifier are copied through exactly as Stage 3
already resolved them; this file never reinterprets a qualifier, never invents
a midpoint for a range, and never turns "balance"/an unparseable token into 0.

── Units ───────────────────────────────────────────────────────────────────────
Stage 3 (as of extract_observations.py's v0.4 property-unit-provenance patch)
now additively carries source_column / reported_unit / unit_verification_status
on every property record. This module surfaces that as-is: *_reported_unit and
*_unit_verified summary columns per target, plus the full per-property fields
preserved unchanged in the audit sidecar. unit_verified is deliberately
narrower than "a unit was confirmed" — it is True only when the confirmed unit
matches the EXPECTED unit for that target column (MPa for UTS/YS, % for
elongation); a genuinely verified GPa or Pa strength value is real, confirmed
information, just not usable in a *_MPa column without a conversion this
module still does not perform anywhere. Older Stage 3 output predating the
unit-provenance patch (e.g. the existing baseline6/final6 files) simply lacks
these keys — read via .get(), producing reported_unit=None/unit_verified=False,
never a crash, never an inferred unit. No unit conversion is performed
anywhere in this file, and this patch does not change which observations
become modelling rows.

── Provenance ─────────────────────────────────────────────────────────────────
Detailed source locations, quoted spans, and full composition records are
written to a separate JSONL audit sidecar keyed by observation_id, never
duplicated into the flat CSV. The flat CSV carries only the summary provenance
columns specified in the approved design (material_identity_status,
processing_identity_status, test_conditions_status, stage3_observation_status,
per-target *_provenance_verified flags, stage3_code_version) — column names
chosen to be directly usable as a
literature_pipeline.modelling.dataset_contract.ProvenanceColumnMap.
Stage 3 LLM confidence is never read or used anywhere in this file (Stage 3
itself never lets it influence gate decisions either — see the Stage 3
forensic notes; this module stays consistent with that).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from literature_pipeline.modelling.dataset_contract import (
    CORE_COMPOSITION_ELEMENTS,
    REQUIRED_IDENTITY_COLUMNS,
    TARGET_COLUMNS,
    composition_column,
    composition_qualifier_column,
)

# ══════════════════════════════════════════════════════════════════════════
# Column layout
# ══════════════════════════════════════════════════════════════════════════

# Stage 3's canonical property keys -> the dataset_contract target column each
# maps to. Asserted against dataset_contract.TARGET_COLUMNS below so silent
# drift between the two modules is caught at import time, not at runtime.
STAGE3_PROPERTY_TO_TARGET_COLUMN = {"UTS": "UTS_MPa", "YS": "YS_MPa", "elongation": "elongation_pct"}
assert set(STAGE3_PROPERTY_TO_TARGET_COLUMN.values()) == set(TARGET_COLUMNS), (
    "STAGE3_PROPERTY_TO_TARGET_COLUMN is out of sync with dataset_contract.TARGET_COLUMNS"
)

METADATA_COLUMNS = ["reported_grade_label", "stage2_evidence_group_ref", "processing_key"]

COMPOSITION_COLUMNS: list[str] = []
for _element in CORE_COMPOSITION_ELEMENTS:
    COMPOSITION_COLUMNS.append(composition_column(_element))
    COMPOSITION_COLUMNS.append(composition_qualifier_column(_element))

PROVENANCE_COLUMNS = [
    "material_identity_status",
    "processing_identity_status",
    "test_conditions_status",
    "stage3_observation_status",
    "UTS_provenance_verified",
    "YS_provenance_verified",
    "elongation_provenance_verified",
    "UTS_reported_unit",
    "YS_reported_unit",
    "elongation_reported_unit",
    "UTS_unit_verified",
    "YS_unit_verified",
    "elongation_unit_verified",
    "elongation_type",
    "stage3_code_version",
]

# Stage 3 keys elongation properties by sub-type ("elongation_total",
# "elongation_uniform", "elongation_fracture", "elongation_unspecified")
# rather than a single bare "elongation", since one observation can
# legitimately carry more than one, genuinely distinct elongation
# definition (e.g. both a uniform- and a total-elongation value) that must
# never be combined as replicates of each other. See map_targets.
_ELONGATION_PROPERTY_PREFIX = "elongation"

# Expected unit for each target column -- MPa for the two strength targets,
# % for elongation. unit_verified is True only when the CONFIRMED unit
# matches this exactly; a verified GPa/Pa value is real information but is
# never marked unit_verified for a *_MPa column, since no conversion happens
# anywhere in this module.
EXPECTED_UNIT_FOR_TARGET_COLUMN = {"UTS_MPa": "MPa", "YS_MPa": "MPa", "elongation_pct": "%"}
VERIFIED_UNIT_STATUSES = {"verified_from_table_header", "verified_from_local_text"}

ALL_COLUMNS = (
    list(REQUIRED_IDENTITY_COLUMNS)
    + METADATA_COLUMNS
    + list(TARGET_COLUMNS)
    + COMPOSITION_COLUMNS
    + PROVENANCE_COLUMNS
)

# Flat-dataset schema version -- recorded only in the QC report sidecar
# (cpp_dataset_v2_qc_report.json), never as a per-row CSV column (every row
# from the same build already shares one schema; a per-row marker would be
# redundant). Bump when ALL_COLUMNS or the semantic meaning of an existing
# column changes (e.g. the elongation_type column added alongside the
# elongation-semantics fix -- 52 -> 53 columns).
DATASET_SCHEMA_VERSION = "cpp_dataset_v2_schema_v0.2_elongation_type"


# ══════════════════════════════════════════════════════════════════════════
# Errors
# ══════════════════════════════════════════════════════════════════════════

class DatasetBuildError(Exception):
    pass


class DuplicateIdentityError(DatasetBuildError):
    """Hard failure: duplicate observation_id, or duplicate
    (doi, material_id, condition_id, observation_id) tuple — see
    audit_duplicates(). Never auto-resolved; the caller must fix the
    upstream Stage 3 output or the build itself."""
    pass


# ══════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════

def load_stage3_record(path: Path) -> dict:
    """Reads one Stage 3 output JSON file. Pure I/O, no transformation."""
    return json.loads(Path(path).read_text())


# ══════════════════════════════════════════════════════════════════════════
# Composition mapping — material level, propagated onto each observation row
# ══════════════════════════════════════════════════════════════════════════

def map_composition(material: dict) -> dict:
    """Returns the flat element/qualifier column dict for one material.

    For each of dataset_contract.CORE_COMPOSITION_ELEMENTS: if Stage 3's
    composition.elements contains a record for that element, its `value`
    (already NaN-safe: None for balance/range/not_reported/unparseable,
    a real float for exact/less_than/greater_than — Stage 3's own resolved
    number, never re-derived here) and `qualifier` are copied through
    unchanged. An element Stage 3 never detected at all for this material
    (not merely unparseable, but absent from composition.elements) gets
    NaN/None — distinct from "not_reported" (Stage 3 saw the cell but
    couldn't parse it) versus "never had a column for this element at all"."""
    elements = {el.get("element"): el for el in (material.get("composition", {}).get("elements") or [])}
    row = {}
    for element in CORE_COMPOSITION_ELEMENTS:
        el = elements.get(element)
        if el is None:
            row[composition_column(element)] = float("nan")
            row[composition_qualifier_column(element)] = None
        else:
            value = el.get("value")
            row[composition_column(element)] = float(value) if value is not None else float("nan")
            row[composition_qualifier_column(element)] = el.get("qualifier")
    return row


# ══════════════════════════════════════════════════════════════════════════
# Target mapping — per observation
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TargetMappingResult:
    values: dict            # {"UTS_MPa": float|nan, ...}
    provenance_verified: dict  # {"UTS_provenance_verified": bool, ...}
    unit_summary: dict      # {"UTS_reported_unit": str|None, "UTS_unit_verified": bool, ...}
    replicate_unresolved_properties: list  # Stage3 canonical names ("UTS","YS","elongation")
                                             # whose value is None because of an
                                             # unresolved replicate cluster
    elongation_type: str | None = None      # "total"|"uniform"|"fracture"|"unspecified" when
                                             # exactly one elongation sub-type is present; None
                                             # if absent OR if more than one type is present
                                             # (see elongation_type_conflict)
    elongation_type_conflict: bool = False  # True when the observation contains MORE THAN ONE
                                             # distinct elongation sub-type -- elongation_pct is
                                             # left unresolved (nan) rather than silently picking
                                             # one or averaging; never a replicate
    incompatible_scope_targets: dict = field(default_factory=dict)  # {target_col: reason_code} for
                                             # a UTS/YS target whose ONLY evidence is positively
                                             # scope/loading-mode incompatible with the
                                             # bulk/macroscale-tensile target (a compressive,
                                             # microscale/local, or phase-level value) -- the
                                             # target's value stays nan (never mapped), but
                                             # provenance_verified stays True (real, verified
                                             # evidence exists, it's just not eligible for THIS
                                             # target); reason_code is "incompatible_property_scope"
                                             # (takes priority when scope is non-unspecified) or
                                             # "incompatible_loading_mode"


def map_targets(observation: dict) -> TargetMappingResult:
    """Maps Stage 3's `properties` dict onto the three target columns.

    Provenance-verified rule (approved design): True whenever the property
    key is present in this valid_observation's `properties` dict at all —
    that presence IS "survived Stage 3's validation path" (only
    provenance-reverified, gate-passed items ever reach `properties`). This
    is independent of whether the resolved numeric value happens to be NaN
    due to an unresolved replicate cluster — replicate ambiguity is a
    downstream selection question, not a provenance failure.

    A property's `value` being None in the current Stage 3 schema only ever
    occurs for a multi-item replicate cluster with no single positively-
    compatible value (see build_valid_observation in extract_observations.py
    — the single-item path always carries a real parsed float). This
    function does not re-verify that assumption; it labels every None value
    it sees as replicate-unresolved, consistent with Stage 3's current,
    only implemented reason for it existing at all.

    Unit summary: reported_unit/unit_verification_status are read via .get()
    so an older Stage 3 record predating the unit-provenance patch (no such
    keys at all) safely produces reported_unit=None / unit_verified=False —
    never a crash, never an inferred unit. unit_verified is computed
    independently of whether `value` itself resolved (same rationale as
    provenance_verified above: unit provenance describes what Stage 3 knows
    about the property, not whether a replicate cluster happened to
    resolve). This function performs NO unit conversion and does not change
    which target values are populated — see build_dataset's module
    docstring."""
    properties = observation.get("properties", {}) or {}
    values, provenance_verified, unit_summary, replicate_unresolved = {}, {}, {}, []
    elongation_type_out, elongation_conflict = None, False
    incompatible_scope_targets: dict = {}

    for stage3_prop, target_col in STAGE3_PROPERTY_TO_TARGET_COLUMN.items():
        if stage3_prop == _ELONGATION_PROPERTY_PREFIX:
            # Stage 3 keys elongation by sub-type (elongation_total,
            # elongation_uniform, elongation_fracture,
            # elongation_unspecified) so a uniform- and a total-elongation
            # value in the SAME observation occupy separate properties-dict
            # slots rather than colliding. Exactly one such key present ->
            # resolve normally, carrying its elongation_type through. More
            # than one present -> genuinely conflicting definitions for one
            # target column; never averaged, never silently picked --
            # elongation_pct is left unresolved for this observation
            # (flagged distinctly from a true unresolved-replicate case).
            matching = {k: v for k, v in properties.items()
                        if k == _ELONGATION_PROPERTY_PREFIX or k.startswith(_ELONGATION_PROPERTY_PREFIX + "_")}
            if len(matching) > 1:
                # Real, verified evidence exists (more than one kind of
                # it) -- this is not "no evidence", so provenance_verified
                # stays True, consistent with how an unresolved-replicate
                # cluster is treated below. Only the numeric target value
                # is left unresolved, because picking one type over
                # another here would be exactly the silent-selection this
                # task must not do.
                elongation_conflict = True
                values[target_col] = float("nan")
                provenance_verified[f"{stage3_prop}_provenance_verified"] = True
                unit_summary[f"{stage3_prop}_reported_unit"] = None
                unit_summary[f"{stage3_prop}_unit_verified"] = False
                continue
            prop_data = next(iter(matching.values())) if matching else None
            if prop_data is not None:
                elongation_type_out = prop_data.get("elongation_type")
        else:
            prop_data = properties.get(stage3_prop)

        if prop_data is None:
            # For UTS/YS only: the plain key is absent, but a scope/mode
            # -qualified variant (e.g. "YS_phase_level_compressive", see
            # extract_observations._scope_qualified_property_key) may still
            # be present -- that means real, Stage-3-verified evidence DOES
            # exist for this property, it is just positively incompatible
            # with the bulk/macroscale-tensile target, a materially
            # different situation from "no evidence was found at all".
            # Never averaged or substituted into the target value; only the
            # reason and provenance_verified reflect the distinction.
            incompatible = None
            if stage3_prop in ("UTS", "YS"):
                qualified = {k: v for k, v in properties.items() if k.startswith(f"{stage3_prop}_")}
                if qualified:
                    q_data = next(iter(qualified.values()))
                    q_scope = q_data.get("property_scope")
                    reason = "incompatible_property_scope" if q_scope not in (None, "unspecified") \
                        else "incompatible_loading_mode"
                    incompatible = reason
            if incompatible is not None:
                incompatible_scope_targets[target_col] = incompatible
                values[target_col] = float("nan")
                provenance_verified[f"{stage3_prop}_provenance_verified"] = True
                unit_summary[f"{stage3_prop}_reported_unit"] = None
                unit_summary[f"{stage3_prop}_unit_verified"] = False
                continue
            values[target_col] = float("nan")
            provenance_verified[f"{stage3_prop}_provenance_verified"] = False
            unit_summary[f"{stage3_prop}_reported_unit"] = None
            unit_summary[f"{stage3_prop}_unit_verified"] = False
            continue
        provenance_verified[f"{stage3_prop}_provenance_verified"] = True

        reported_unit = prop_data.get("reported_unit")
        unit_status = prop_data.get("unit_verification_status")
        expected_unit = EXPECTED_UNIT_FOR_TARGET_COLUMN[target_col]
        unit_summary[f"{stage3_prop}_reported_unit"] = reported_unit
        unit_summary[f"{stage3_prop}_unit_verified"] = bool(
            reported_unit == expected_unit and unit_status in VERIFIED_UNIT_STATUSES)

        value = prop_data.get("value")
        if value is None:
            values[target_col] = float("nan")
            replicate_unresolved.append(stage3_prop)
        else:
            values[target_col] = float(value)
    return TargetMappingResult(values=values, provenance_verified=provenance_verified,
                                unit_summary=unit_summary, replicate_unresolved_properties=replicate_unresolved,
                                elongation_type=elongation_type_out, elongation_type_conflict=elongation_conflict,
                                incompatible_scope_targets=incompatible_scope_targets)


# ══════════════════════════════════════════════════════════════════════════
# Provenance summary mapping
# ══════════════════════════════════════════════════════════════════════════

def map_provenance(observation: dict, stage3_record: dict, target_provenance_verified: dict) -> dict:
    linkage = observation.get("linkage", {}) or {}
    row = {
        "material_identity_status": linkage.get("material_identity"),
        "processing_identity_status": linkage.get("processing_identity"),
        "test_conditions_status": linkage.get("test_conditions_status"),
        "stage3_observation_status": observation.get("observation_status"),
        "stage3_code_version": stage3_record.get("stage3_code_version"),
    }
    row.update({
        "UTS_provenance_verified": target_provenance_verified.get("UTS_provenance_verified", False),
        "YS_provenance_verified": target_provenance_verified.get("YS_provenance_verified", False),
        "elongation_provenance_verified": target_provenance_verified.get("elongation_provenance_verified", False),
    })
    return row


# ══════════════════════════════════════════════════════════════════════════
# Per-observation flattening
# ══════════════════════════════════════════════════════════════════════════

def _has_usable_target(target_values: dict) -> bool:
    return any(not math.isnan(v) for v in target_values.values())


def flatten_observation(doi: str, material: dict, condition: dict, observation: dict, stage3_record: dict) -> dict:
    """Builds the full flat row for one valid_observation, regardless of
    whether it ends up usable (that decision is flatten_stage3_record's
    job, made uniformly alongside every other exclusion reason). Always
    returns every column in ALL_COLUMNS."""
    target_result = map_targets(observation)
    row = {
        "doi": doi,
        "material_id": material.get("material_id"),
        "condition_id": condition.get("condition_id"),
        "observation_id": observation.get("observation_id"),
        "reported_grade_label": material.get("reported_grade_label"),
        "stage2_evidence_group_ref": observation.get("stage2_evidence_group_ref"),
        "processing_key": condition.get("processing_key"),
    }
    row.update(target_result.values)
    row.update(map_composition(material))
    row.update(map_provenance(observation, stage3_record, target_result.provenance_verified))
    row.update(target_result.unit_summary)
    # None whenever elongation_pct itself is unresolved (absent, unresolved
    # replicate, or a multi-type conflict) -- never guessed, never carried
    # over from a different observation.
    row["elongation_type"] = target_result.elongation_type
    return row


def _audit_record(doi: str, material: dict, condition: dict, observation: dict) -> dict:
    """Full detailed provenance for one included row, keyed by
    observation_id — table/row/column locations, quoted spans, raw
    replicate values, and the material's complete composition record
    (including reported_value/reported_unit/range_low/range_high/provenance
    per element), none of which are duplicated into the flat CSV."""
    return {
        "observation_id": observation.get("observation_id"),
        "doi": doi,
        "material_id": material.get("material_id"),
        "condition_id": condition.get("condition_id"),
        "stage2_evidence_group_ref": observation.get("stage2_evidence_group_ref"),
        "processing_key": condition.get("processing_key"),
        "full_composition": material.get("composition"),
        "linkage": observation.get("linkage"),
        "property_completeness": observation.get("property_completeness"),
        "properties": observation.get("properties"),
    }


# ══════════════════════════════════════════════════════════════════════════
# Per-paper flattening — walks materials -> processing_conditions ->
# observations, and separately collects everything Stage 3 already excluded
# (orphan/rejected/failures) plus anything THIS module excludes
# (replicate_target_unresolved).
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FlattenResult:
    rows: list          # included modelling rows (dicts)
    audit_records: list  # one per included row
    excluded_records: list  # every excluded record, tagged with exclusion_source + reason


def flatten_stage3_record(stage3_record: dict) -> FlattenResult:
    doi = stage3_record.get("doi")
    rows, audit_records, excluded = [], [], []

    for material in stage3_record.get("materials", []):
        for condition in material.get("processing_conditions", []):
            for observation in condition.get("observations", []):
                if observation.get("observation_status") != "valid_observation":
                    # Stage 3's own hierarchy is not expected to place anything
                    # but valid_observation under processing_conditions[*].observations
                    # (orphans/rejections live in the paper-level lists below), but
                    # this check is kept as a defensive, non-silent guard rather
                    # than assuming that invariant holds forever.
                    excluded.append({
                        "doi": doi, "material_id": material.get("material_id"),
                        "condition_id": condition.get("condition_id"),
                        "observation_id": observation.get("observation_id"),
                        "exclusion_source": "unexpected_non_valid_observation_in_hierarchy",
                        "reason": observation.get("observation_status"),
                        "detail": observation,
                    })
                    continue

                row = flatten_observation(doi, material, condition, observation, stage3_record)
                target_result = map_targets(observation)  # re-derived for the exclusion-reason check only
                if _has_usable_target(target_result.values):
                    rows.append(row)
                    audit_records.append(_audit_record(doi, material, condition, observation))
                else:
                    excluded.append({
                        "doi": doi, "material_id": material.get("material_id"),
                        "condition_id": condition.get("condition_id"),
                        "observation_id": observation.get("observation_id"),
                        "exclusion_source": "valid_observation_no_usable_target",
                        "reason": (
                            "elongation_type_conflict_unresolved" if target_result.elongation_type_conflict
                            else next(iter(target_result.incompatible_scope_targets.values()))
                            if target_result.incompatible_scope_targets
                            else "replicate_target_unresolved" if target_result.replicate_unresolved_properties
                            else "no_usable_target"
                        ),
                        "detail": observation,
                    })

    for orphan in stage3_record.get("unresolved_candidates", []):
        excluded.append({
            "doi": doi, "material_id": orphan.get("material_id"), "condition_id": None,
            "observation_id": None, "exclusion_source": "orphan_property_evidence",
            "reason": orphan.get("reason"), "detail": orphan,
        })
    for rejected in stage3_record.get("rejected_candidates", []):
        excluded.append({
            "doi": doi, "material_id": rejected.get("material_id"), "condition_id": None,
            "observation_id": None, "exclusion_source": "rejected_candidate",
            "reason": rejected.get("reason"), "detail": rejected,
        })
    for failure in stage3_record.get("failures", []):
        excluded.append({
            "doi": doi, "material_id": failure.get("material_id"), "condition_id": None,
            "observation_id": None, "exclusion_source": "failure",
            "reason": failure.get("type"), "detail": failure,
        })

    return FlattenResult(rows=rows, audit_records=audit_records, excluded_records=excluded)


# ══════════════════════════════════════════════════════════════════════════
# Duplicate audit
# ══════════════════════════════════════════════════════════════════════════

def audit_duplicates(df: pd.DataFrame, audit_records: list) -> dict:
    """HARD-FAIL checks (raise DuplicateIdentityError immediately, listing
    every violation) vs. WARNING-only checks (returned in the report,
    never auto-deleted — per approved design, ambiguous cases are surfaced
    for manual review, not resolved automatically).

    Location-based warning checks use `audit_records` (not the flat df)
    since table/row locations live only in the audit sidecar data, not in
    the flat columns. A property located in prose (no row_key — Stage 3's
    source_context does not carry a paragraph index for prose-sourced
    property items) cannot be location-compared here; this is a stated
    limitation, not a silent gap — prose-sourced properties are simply
    excluded from the shared-location check, never falsely flagged or
    falsely cleared."""
    hard_failures = []

    # Scoped to (doi, observation_id), not bare observation_id: Stage 3's own
    # ID generation (condition_id = f"{material_id}-P{n}", observation_id =
    # f"{condition_id}-O{n}", see assemble_processing_conditions in
    # extract_observations.py) provides NO cross-paper uniqueness guarantee
    # or intention at all -- two entirely unrelated papers routinely and
    # legitimately both produce "M1-P1-O1". A bare-string global check would
    # hard-fail on every ordinary multi-paper build for a reason that is not
    # actually a defect. Within one DOI, Stage 3's own ID scheme guarantees
    # observation_id uniqueness by construction (material_id is embedded in
    # every descendant ID), so a same-DOI collision here is always a genuine
    # upstream anomaly worth failing hard on.
    dup_obs_id_df = df[df.duplicated(subset=["doi", "observation_id"], keep=False)]
    dup_obs_id = dup_obs_id_df[["doi", "observation_id"]].drop_duplicates().to_dict("records")
    if dup_obs_id:
        hard_failures.append(f"duplicate observation_id within the same doi: {dup_obs_id}")

    identity_cols = ["doi", "material_id", "condition_id", "observation_id"]
    dup_tuple_mask = df.duplicated(subset=identity_cols, keep=False)
    dup_tuples = df.loc[dup_tuple_mask, identity_cols].drop_duplicates().to_dict("records")
    if dup_tuples:
        hard_failures.append(f"duplicate (doi, material_id, condition_id, observation_id) tuples: {dup_tuples}")

    if hard_failures:
        raise DuplicateIdentityError("; ".join(hard_failures))

    non_id_cols = [c for c in ALL_COLUMNS if c not in identity_cols]
    dup_scientific_mask = df.duplicated(subset=non_id_cols, keep=False)
    duplicate_scientific_rows = df.loc[dup_scientific_mask, identity_cols].to_dict("records")

    location_index: dict = {}
    shared_source_location, shared_table_row_property = [], []
    for rec in audit_records:
        obs_id = rec["observation_id"]
        for prop_name, prop_data in (rec.get("properties") or {}).items():
            sc = (prop_data or {}).get("source_context") or {}
            row_key = sc.get("row_key")
            if row_key is None:
                continue  # prose-sourced or otherwise un-locatable — see docstring
            key = (rec["doi"], prop_name, tuple(row_key))
            location_index.setdefault(key, []).append(obs_id)

    for (doi, prop_name, row_key), obs_ids in location_index.items():
        if len(obs_ids) > 1:
            entry = {"doi": doi, "property": prop_name, "row_key": list(row_key), "observation_ids": obs_ids}
            shared_source_location.append(entry)
            shared_table_row_property.append(entry)

    return {
        "duplicate_observation_id": [],
        "duplicate_identity_tuple": [],
        "duplicate_scientific_rows": duplicate_scientific_rows,
        "shared_source_location": shared_source_location,
        "shared_table_row_property": shared_table_row_property,
    }


# ══════════════════════════════════════════════════════════════════════════
# QC checks — soft, scientific warnings only. Never raises, never modifies df.
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_EXTREME_STRENGTH_MPA = 3000.0
_MAX_UNIT_COVERAGE_EXAMPLES = 5


def _unit_coverage_report(df: pd.DataFrame, audit_records: list) -> dict:
    """Real per-target unit-coverage counts, replacing the old blanket
    target_unit_verification_status string. Uses audit_records (not just the
    flat df) because the flat table only carries reported_unit/unit_verified
    — distinguishing "Stage 3 explicitly could not confirm a claimed unit"
    (reported_by_model_unverified) from "no unit information at all"
    (not_available, or an older pre-patch record entirely missing these
    keys) requires unit_verification_status, which lives only in the audit
    sidecar's full property records, by design (see the module docstring's
    Provenance section on why detailed provenance isn't duplicated into the
    flat CSV). Never rejects a row — purely descriptive."""
    audit_by_obs_id = {rec["observation_id"]: rec for rec in audit_records}
    report = {}
    for stage3_prop, target_col in STAGE3_PROPERTY_TO_TARGET_COLUMN.items():
        expected_unit = EXPECTED_UNIT_FOR_TARGET_COLUMN[target_col]
        n_target_values = 0
        n_expected_unit_verified = 0
        n_verified_other_unit = 0
        n_unit_unverified = 0
        n_unit_not_available = 0
        examples_other_unit, examples_unverified, examples_not_available = [], [], []

        for _, df_row in df.iterrows():
            if pd.isna(df_row[target_col]):
                continue
            n_target_values += 1
            obs_id = df_row["observation_id"]
            audit_rec = audit_by_obs_id.get(obs_id) or {}
            prop_data = (audit_rec.get("properties") or {}).get(stage3_prop) or {}
            reported_unit = prop_data.get("reported_unit")
            status = prop_data.get("unit_verification_status")

            if reported_unit == expected_unit and status in VERIFIED_UNIT_STATUSES:
                n_expected_unit_verified += 1
            elif reported_unit is not None and status in VERIFIED_UNIT_STATUSES:
                n_verified_other_unit += 1
                if len(examples_other_unit) < _MAX_UNIT_COVERAGE_EXAMPLES:
                    examples_other_unit.append(obs_id)
            elif status == "reported_by_model_unverified":
                n_unit_unverified += 1
                if len(examples_unverified) < _MAX_UNIT_COVERAGE_EXAMPLES:
                    examples_unverified.append(obs_id)
            else:
                # status == "not_available", or no audit record / no such
                # property key at all (older, pre-unit-provenance-patch
                # Stage 3 output, e.g. the existing baseline6/final6 files).
                n_unit_not_available += 1
                if len(examples_not_available) < _MAX_UNIT_COVERAGE_EXAMPLES:
                    examples_not_available.append(obs_id)

        report[target_col] = {
            "n_target_values": n_target_values,
            "n_expected_unit_verified": n_expected_unit_verified,
            "n_verified_other_unit": n_verified_other_unit,
            "n_unit_unverified": n_unit_unverified,
            "n_unit_not_available": n_unit_not_available,
            "example_observation_ids_verified_other_unit": examples_other_unit,
            "example_observation_ids_unverified_unit_claim": examples_unverified,
            "example_observation_ids_unit_not_available": examples_not_available,
        }
    return report


def run_qc_checks(df: pd.DataFrame, audit_records: list | None = None,
                   extreme_strength_threshold_mpa: float = DEFAULT_EXTREME_STRENGTH_MPA) -> dict:
    warnings_list = []

    def _flag(mask, message):
        ids = df.loc[mask, "observation_id"].tolist()
        if ids:
            warnings_list.append({"check": message, "observation_ids": ids, "count": len(ids)})

    both_present = df["UTS_MPa"].notna() & df["YS_MPa"].notna()
    _flag(both_present & (df["UTS_MPa"] < df["YS_MPa"]), "UTS_MPa < YS_MPa")
    _flag(df["UTS_MPa"].notna() & (df["UTS_MPa"] < 0), "negative UTS_MPa")
    _flag(df["YS_MPa"].notna() & (df["YS_MPa"] < 0), "negative YS_MPa")
    _flag(df["elongation_pct"].notna() & (df["elongation_pct"] < 0), "elongation_pct < 0")
    _flag(df["elongation_pct"].notna() & (df["elongation_pct"] > 100), "elongation_pct > 100")
    _flag(df["UTS_MPa"].notna() & (df["UTS_MPa"] > extreme_strength_threshold_mpa),
          f"UTS_MPa above configurable threshold ({extreme_strength_threshold_mpa} MPa)")
    _flag(df["YS_MPa"].notna() & (df["YS_MPa"] > extreme_strength_threshold_mpa),
          f"YS_MPa above configurable threshold ({extreme_strength_threshold_mpa} MPa)")

    element_cols = [composition_column(e) for e in CORE_COMPOSITION_ELEMENTS]
    composition_sum = df[element_cols].sum(axis=1, skipna=True)
    _flag(composition_sum > 100.0, "sum of reported composition elements exceeds 100 wt.%")

    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "target_unit_coverage": _unit_coverage_report(df, audit_records or []),
        "extreme_strength_threshold_mpa": extreme_strength_threshold_mpa,
        "n_rows_checked": int(len(df)),
        "warnings": warnings_list,
    }


# ══════════════════════════════════════════════════════════════════════════
# Top-level build orchestration
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BuildResult:
    dataset: pd.DataFrame
    audit_records: list
    excluded_records: list
    duplicate_report: dict
    qc_report: dict


def build_dataset(stage3_paths: list) -> BuildResult:
    """stage3_paths: an iterable of Path (or str) to individual Stage 3 JSON
    files — one per paper. Not a directory glob itself, so the caller
    controls exactly which files are included (e.g. excluding known
    non-final/experimental runs)."""
    all_rows, all_audit, all_excluded = [], [], []
    for p in stage3_paths:
        record = load_stage3_record(Path(p))
        result = flatten_stage3_record(record)
        all_rows += result.rows
        all_audit += result.audit_records
        all_excluded += result.excluded_records

    dataset = pd.DataFrame(all_rows, columns=ALL_COLUMNS)
    duplicate_report = audit_duplicates(dataset, all_audit)
    # NOTE (incidental simplification, not a behaviour change): run_qc_checks
    # was previously bypassed for an empty dataset via a hand-written
    # fallback dict, because target_unit_verification_status was a single
    # constant string that made sense to hardcode either way. The new
    # per-target _unit_coverage_report already produces correct all-zero
    # counts for a 0-row DataFrame on its own (confirmed: every check in
    # this function operates safely on an empty df/mask), so that special
    # case was removed rather than duplicated -- run_qc_checks(dataset, ...)
    # now runs unconditionally and produces one consistent report shape
    # whether the dataset has rows or not.
    qc_report = run_qc_checks(dataset, all_audit)

    return BuildResult(dataset=dataset, audit_records=all_audit, excluded_records=all_excluded,
                        duplicate_report=duplicate_report, qc_report=qc_report)


# ══════════════════════════════════════════════════════════════════════════
# Output writing — production runs only. Tests must pass a temporary
# directory, never this default.
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_OUT_DIR = Path("data/datasets_v2")


def _excluded_summary(excluded_records: list) -> pd.DataFrame:
    rows = []
    for rec in excluded_records:
        rows.append({"doi": rec.get("doi"), "material_id": rec.get("material_id"),
                      "condition_id": rec.get("condition_id"), "observation_id": rec.get("observation_id"),
                      "exclusion_source": rec.get("exclusion_source"), "reason": rec.get("reason")})
    return pd.DataFrame(rows, columns=["doi", "material_id", "condition_id", "observation_id",
                                        "exclusion_source", "reason"])


def write_dataset(result: BuildResult, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    """Writes the five approved output files under out_dir. Never touches
    Stage 3 output — out_dir defaults to data/datasets_v2/, never
    data/extracted_v2/. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "dataset": out_dir / "cpp_dataset_v2.csv",
        "audit": out_dir / "cpp_dataset_v2_audit.jsonl",
        "qc_report": out_dir / "cpp_dataset_v2_qc_report.json",
        "duplicate_report": out_dir / "cpp_dataset_v2_duplicate_report.json",
        "excluded_summary": out_dir / "cpp_dataset_v2_excluded_summary.csv",
    }

    result.dataset.to_csv(paths["dataset"], index=False)
    with open(paths["audit"], "w") as f:
        for rec in result.audit_records:
            f.write(json.dumps(rec, default=str) + "\n")
    with open(paths["qc_report"], "w") as f:
        json.dump(result.qc_report, f, indent=2, default=str)
    with open(paths["duplicate_report"], "w") as f:
        json.dump(result.duplicate_report, f, indent=2, default=str)
    _excluded_summary(result.excluded_records).to_csv(paths["excluded_summary"], index=False)

    return paths
