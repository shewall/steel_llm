"""
dataset_contract.py — passive schema description + validation for the final
CPP predictive-modelling dataset.

Deliberately independent of sklearn/scipy (pure pandas + dataclasses) so it
stays cheap to import from anywhere, including a future Stage 5 dataset-
construction script.

── What is intentionally NOT frozen here (and why) ───────────────────────────
  - Which literal enum values count as "confirmed" material/processing
    identity, or which column names carry provenance status in the real
    final dataset — callers supply a ProvenanceColumnMap and assemble
    QualityTierRule predicates from the primitives below. Nothing here
    hardcodes a status string.
  - Composition qualifier -> numeric encoding (trace/balance/range/
    approximate) — lives in modelling_lib.py's resolve_composition_column,
    not here; this module only knows the element/qualifier COLUMN NAMES.
  - Replicate aggregation policy — removed from the modelling library
    entirely (see modelling_lib.py's module docstring). This module carries
    no replicate-related fields at all.
  - The full flattened processing-feature column list — feature lists are
    always caller-supplied at call time, never enumerated here.

── What IS fixed here ────────────────────────────────────────────────────────
  - The minimum identity columns.
  - The three target column NAMES (UTS_MPa, YS_MPa, elongation_pct) — all
    three are REQUIRED to exist as columns; their VALUES remain
    independently nullable per observation.
  - The grouping column (doi).
  - The core composition element set, as COLUMNS only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

REQUIRED_IDENTITY_COLUMNS = ["doi", "material_id", "condition_id", "observation_id"]
TARGET_COLUMNS = ["UTS_MPa", "YS_MPa", "elongation_pct"]
GROUP_COLUMN = "doi"
CORE_COMPOSITION_ELEMENTS = ["C", "Mn", "Si", "Al", "Cr", "Ni", "Mo", "Nb", "Ti", "V", "Cu", "P", "S", "N"]


def composition_column(element: str) -> str:
    return element


def composition_qualifier_column(element: str) -> str:
    return f"{element}_qualifier"


@dataclass
class ProvenanceColumnMap:
    """Caller-supplied mapping from CONCEPTUAL provenance roles to actual
    column names in whatever the real final dataset turns out to call them.
    stage2_version_column is explicitly optional — it is retained only as
    optional provenance metadata / for a separate Stage2-v1-vs-v2 validation
    dataset, not a mandatory modelling-column requirement."""
    material_identity_status_column: str
    processing_identity_status_column: str
    target_provenance_verified_column: dict[str, str]  # {"UTS_MPa": "...", ...}
    stage3_observation_status_column: str | None = None
    stage2_version_column: str | None = None


@dataclass
class QualityTierRule:
    """A named, fully caller-defined filter rule. Only the SHAPE (a named
    predicate over a dataframe) is fixed — which columns/values it checks
    is assembled by the caller from the predicate builders below."""
    name: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def make_status_in_predicate(column: str, allowed_values: Iterable) -> Callable[[pd.DataFrame], pd.Series]:
    """Predicate: df[column] is one of allowed_values. If column doesn't
    exist, the predicate is False everywhere (never raises at predicate-
    application time) — missing-column detection belongs to
    validate_minimum_structure, not to silent predicate evaluation."""
    allowed = set(allowed_values)

    def _predicate(df: pd.DataFrame) -> pd.Series:
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        return df[column].isin(allowed)

    return _predicate


def make_target_has_value_predicate(target: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Predicate: df[target] is non-null. The base building block for
    target-specific tiers — a row missing YS/elongation must not disqualify
    an otherwise-usable UTS observation, so each tier is built per-target
    starting from THIS predicate, never from "all three targets present"."""

    def _predicate(df: pd.DataFrame) -> pd.Series:
        if target not in df.columns:
            return pd.Series(False, index=df.index)
        return df[target].notna()

    return _predicate


def make_target_provenance_predicate(column_map: dict[str, str], targets: Iterable[str] | None = None) -> Callable[[pd.DataFrame], pd.Series]:
    """Predicate: provenance-verified == True for the given targets ONLY
    (defaults to every target in column_map if targets is not given).
    Callers building a target-specific high-confidence tier MUST pass
    targets=[<the one target>] so a high-confidence UTS tier never
    accidentally requires YS/elongation provenance verification too."""
    keys = list(column_map.keys()) if targets is None else list(targets)

    def _predicate(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for t in keys:
            col = column_map.get(t)
            if col is None or col not in df.columns:
                mask &= False
                continue
            mask &= df[col].fillna(False).astype(bool)
        return mask

    return _predicate


def make_and_predicate(*predicates: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
    """Combines predicates with logical AND — how a caller assembles
    'confirmed material identity AND confirmed processing identity AND
    target provenance verified' without that combination being hardcoded
    anywhere in this module."""

    def _predicate(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for p in predicates:
            mask &= p(df)
        return mask

    return _predicate


@dataclass
class ValidationReport:
    ok: bool
    missing_required_columns: list[str]
    missing_target_columns: list[str]
    has_any_target_data: bool
    missing_provenance_columns: list[str]
    warnings: list[str]


def validate_minimum_structure(df: pd.DataFrame, provenance_map: ProvenanceColumnMap | None = None) -> ValidationReport:
    """Checks:
      - all REQUIRED_IDENTITY_COLUMNS present
      - all three TARGET_COLUMNS present (values may still be all-null per
        column; that is caught separately by has_any_target_data)
      - at least one target column has at least one non-null value
        somewhere in the dataframe
      - if provenance_map is given, every column it references exists
        (provenance_map is optional — this validator is usable before
        provenance columns are finalized)
    Returns a report object rather than raising, so callers decide how to
    react (fail hard vs. proceed with a warning)."""
    missing_required = [c for c in REQUIRED_IDENTITY_COLUMNS if c not in df.columns]
    missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]
    warnings: list[str] = []

    if missing_targets:
        has_any_target_data = False
        warnings.append(f"target columns missing from dataframe: {missing_targets}")
    else:
        has_any_target_data = bool(df[TARGET_COLUMNS].notna().any().any())
        if not has_any_target_data:
            warnings.append("all three target columns are present but contain no non-null values anywhere")

    missing_provenance: list[str] = []
    if provenance_map is not None:
        candidate_cols = [provenance_map.material_identity_status_column,
                           provenance_map.processing_identity_status_column]
        candidate_cols += list(provenance_map.target_provenance_verified_column.values())
        if provenance_map.stage3_observation_status_column:
            candidate_cols.append(provenance_map.stage3_observation_status_column)
        if provenance_map.stage2_version_column:
            candidate_cols.append(provenance_map.stage2_version_column)
        missing_provenance = [c for c in candidate_cols if c not in df.columns]

    ok = (not missing_required) and (not missing_targets) and has_any_target_data and (not missing_provenance)
    return ValidationReport(
        ok=ok, missing_required_columns=missing_required, missing_target_columns=missing_targets,
        has_any_target_data=has_any_target_data, missing_provenance_columns=missing_provenance, warnings=warnings,
    )
