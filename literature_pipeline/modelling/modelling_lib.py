"""
modelling_lib.py — active experiment machinery for the CPP predictive-
modelling skeleton: preprocessing, GroupKFold, the four-model comparison,
metric aggregation, target-specific quality tiers, and (optional, for
later) bootstrap uncertainty.

Imports literature_pipeline.modelling.dataset_contract for the shared
constants/types; owns everything sklearn-dependent.

── What is intentionally NOT implemented here ────────────────────────────────
  - Replicate aggregation. There is deliberately NO resolve_target_value()
    or any function that combines raw_replicate_values into a scalar.
    Replicate aggregation is a Stage-5/final-dataset-construction decision,
    not a modelling decision — this library consumes the final observation-
    level target value exactly as supplied by the final dataset. Replicate
    metadata may exist in that dataset for auditing; nothing here reads it.
  - Any literal "confirmed" enum value or provenance column name — quality
    tiers are assembled from dataset_contract's predicate primitives using
    a caller-supplied ProvenanceColumnMap, never hardcoded.
  - Hyperparameter tuning. Model factories use fixed, reasonable defaults
    with explicit random seeds — no search/CV-inside-CV yet.

── Reproducibility ────────────────────────────────────────────────────────────
All stochastic models (RandomForestRegressor, GradientBoostingRegressor) are
constructed with an explicit random_state (default 42, overridable). Ridge
is deterministic for the solvers used here; random_state is still threaded
through for API uniformity and is harmless. SVR has no random_state
parameter and does not receive one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from literature_pipeline.modelling.dataset_contract import (
    GROUP_COLUMN,
    QualityTierRule,
    ProvenanceColumnMap,
    make_and_predicate,
    make_status_in_predicate,
    make_target_has_value_predicate,
    make_target_provenance_predicate,
)

DEFAULT_RANDOM_STATE = 42

try:
    from sklearn.metrics import root_mean_squared_error as _root_mean_squared_error

    def _rmse(y_true, y_pred) -> float:
        return float(_root_mean_squared_error(y_true, y_pred))
except ImportError:  # older scikit-learn without root_mean_squared_error
    from sklearn.metrics import mean_squared_error as _mean_squared_error

    def _rmse(y_true, y_pred) -> float:
        return float(_mean_squared_error(y_true, y_pred, squared=False))


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn uses `sparse`, not `sparse_output`
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ══════════════════════════════════════════════════════════════════════════
# KNOWN DATA CORRECTIONS / EXCLUSIONS — findings from 01_dataset_qc.ipynb's
# row-by-row audit-trail investigation of cpp_dataset_v2.csv (the 871-paper
# production run). Every (doi, observation_id) pair below was individually
# traced to its source_column / reported_unit / table_caption in
# cpp_dataset_v2_audit.jsonl — none of this is inferred from magnitude alone.
# Lives here (not duplicated in 01 or re-derived in 02) so every notebook
# that touches this dataset applies the identical, single-source rule set.
# ══════════════════════════════════════════════════════════════════════════

# CONFIRMED wrong property/unit — the source evidence directly contradicts
# the column the value was placed in (wrong table column, or a unit
# fundamentally incompatible with the claimed property). Excluded from
# EVERY cohort, including sensitivity — keeping proven-corrupted values in
# a "robustness" cohort would not test anything.
CONFIRMED_WRONG_ELONGATION_KEYS = {
    ("10.1016/j.msea.2026.150164", "M1-P1-O1"),   # "Area Uniform EL (MPa-%)" -- stress-strain-curve area, not elongation
    ("10.1016/j.msea.2026.150164", "M1-P1-O2"),
    ("10.1016/j.msea.2026.150164", "M1-P1-O3"),
    ("10.1016/j.msea.2026.150164", "M1-P1-O4"),
    ("10.1016/j.msea.2026.150164", "M2-P1-O1"),
    ("10.1016/j.msea.2026.150164", "M2-P1-O2"),
    ("10.1016/j.msea.2026.150164", "M2-P1-O3"),
    ("10.1016/j.msea.2026.150164", "M2-P1-O4"),
    ("10.1016/j.heliyon.2024.e32101", "M1-P1-O2"),  # source_column "Average thickness" (mm), not elongation
    ("10.1007/s11661-025-08095-7", "M1-P1-O1"),     # source_column "AGS" from a composition table, not elongation
    ("10.1007/s11661-025-08095-7", "M2-P1-O1"),
    ("10.1016/j.matdes.2025.114234", "M1-P1-O7"),   # elongation_total reported in GPa -- unit fundamentally incompatible with elongation
}

# CANNOT DECIDE — value carries no captured unit and no source_column at
# all (unit_verification_status in {"not_available", "reported_by_model_unverified"}),
# so there is no way to confirm even the PROPERTY is correct, let alone the
# unit. Excluded from every cohort for the same reason as CONFIRMED_WRONG
# (absence of grounding, not presence of contradicting evidence) but kept
# as a separate, honestly-labelled category rather than merged into it.
CANNOT_DECIDE_KEYS = {
    "elongation_pct": {
        ("10.1016/j.corcom.2026.04.005", "M1-P1-O5"),
        ("10.1016/j.jalmes.2025.100214", "M2-P1-O4"),
        ("10.1016/j.matdes.2025.114234", "M1-P1-O4"),
        ("10.1016/j.matdes.2025.114234", "M1-P1-O5"),
    },
    "UTS_MPa": {
        ("10.1016/j.corcom.2026.04.005", "M1-P1-O4"),
    },
    "YS_MPa": {
        ("10.1016/j.jmrt.2022.11.070", "M2-P1-O1"),
        ("10.1007/j.jmrt.2023.10.233", "M5-P1-O1"),  # NOTE: real doi is 10.1016/j.jmrt.2023.10.233, kept exact below
    },
}
# Fix the doi typo above at import time rather than risk a silent mismatch.
CANNOT_DECIDE_KEYS["YS_MPa"] = {
    ("10.1016/j.jmrt.2022.11.070", "M2-P1-O1"),
    ("10.1016/j.jmrt.2023.10.233", "M5-P1-O1"),
}

# CONFIRMED inconsistent, PROBABLE cause only — UTS < YS for the same
# observation is a confirmed physical impossibility (UTS is by definition
# >= YS in a standard tensile test), but the root cause (most likely a
# table row-alignment error; both values pass unit_verified=True with
# source_column "Ultimate Tensile Strength Rm [MPa]" / "Yield Strength
# Rp0.2 [MPa]") has NOT been verified against the real source table.
# Affects BOTH UTS_MPa and YS_MPa for the same two rows. Excluded from
# BOTH primary and sensitivity -- NOT merely "sensitivity-only" material,
# since the inconsistency itself (not just the cause) is confirmed. Only
# appears in a separate, explicitly-labelled diagnostic cohort.
CONFIRMED_INCONSISTENT_PROBABLE_CAUSE_KEYS = {
    ("10.1016/j.ijmecsci.2025.110565", "M1-P1-O7"),
    ("10.1016/j.ijmecsci.2025.110565", "M1-P1-O12"),
}

# CONFIRMED correctable — the source text explicitly states the value in a
# different but unambiguous unit (GPa), so a deterministic scale correction
# is applied, auditably, rather than discarding real information.
UTS_GPA_TO_MPA_CORRECTIONS = {
    ("10.1016/j.jmrt.2021.07.045", "M1-P1-O1"): 1.2,  # confirmed via source text: "1.2 GPa"
}


# CONFIRMED non-steel scope -- found via a systematic title/grade-label audit of all 51 DOIs
# contributing to the final dataset (prompted by 03_model_comparison.ipynb's Ridge
# worst-residual investigation surfacing the first case). Each entry is a (doi, material_id)
# pair whose MATERIAL is not steel at all -- these are full-row exclusions from every steel
# analysis/modelling cohort, not composition corrections, because the material itself is out
# of scope, not merely a mis-extracted value. Source evidence for each, from the paper's own
# title and/or reported_grade_label (never inferred from composition magnitude alone):
#   - 10.1007/s40962-026-02006-1 (M1 "S1", M3 "S3"): title "...AlSi7Mg0.3 (A356) Casting
#     Alloy" -- an aluminium casting alloy. (First case found, via Ridge's extrapolation.)
#   - 10.1007/s10973-026-15761-4 (M1 "AA7075"): title "...7075-alloy..." -- AA7075 is a
#     wrought aluminium alloy designation (Aluminum Association), not steel.
#   - 10.1007/s11665-026-14678-1 (M1 "IN718"): title "...Cold-Sprayed Inconel 718 Coatings"
#     -- Inconel 718 is a nickel-based superalloy.
#   - 10.1007/s40830-026-00604-w (M3 "NiTi (HT)"): title "...Wrought NiTi Shape Memory
#     Alloys" -- nickel-titanium, not an iron-based alloy.
#   - 10.1007/s40962-025-01715-3 (M1 "Al-Mg"): title "Development of New Aluminium
#     Alloys..." -- explicitly an aluminium alloy paper.
#   - 10.1007/s42114-025-01547-z (M5 "CR-A673K"): title "...CuCrZr/Cu laminated metallic
#     composites..." -- a copper-based composite, not steel.
#   - 10.1007/s40194-026-02341-1 (M3 "Cu-foil"): title "Method to systematically investigate
#     the effect of tramp elements in resistance spot welding" -- this material is explicitly
#     labelled "Cu-foil", i.e. the copper foil interlayer/tramp-element insert used in the
#     weld study, not the steel base metal being welded. Its "Cu=999" value (the decimal-comma
#     misparse of "99,9" already identified in 01_dataset_qc.ipynb) is very plausibly a
#     genuine ~99.9 wt.% Cu foil composition -- consistent with the material actually being
#     copper, not corrupted steel data. The paper's own steel base-metal material(s), if
#     extracted, are unaffected by this exclusion (only M3 is excluded); none reached the
#     final primary cohort in the current corpus run.
NON_STEEL_SCOPE_EXCLUSIONS = {
    ("10.1007/s40962-026-02006-1", "M1"),
    ("10.1007/s40962-026-02006-1", "M3"),
    ("10.1007/s10973-026-15761-4", "M1"),
    ("10.1007/s11665-026-14678-1", "M1"),
    ("10.1007/s40830-026-00604-w", "M3"),
    ("10.1007/s40962-025-01715-3", "M1"),
    ("10.1007/s42114-025-01547-z", "M5"),
    ("10.1007/s40194-026-02341-1", "M3"),
}


def non_steel_scope_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows matching a NON_STEEL_SCOPE_EXCLUSIONS (doi, material_id)
    key -- exposed separately (not just inside exclude_non_steel_scope) so
    callers can preserve/report the excluded rows for audit purposes,
    per the "do not delete or alter the raw extraction records, preserve
    excluded records in the audit output" requirement."""
    if "material_id" not in df.columns:
        return pd.Series(False, index=df.index)
    pairs = set(NON_STEEL_SCOPE_EXCLUSIONS)
    return df.apply(lambda r: (r["doi"], r["material_id"]) in pairs, axis=1)


def exclude_non_steel_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a COPY of df with every NON_STEEL_SCOPE_EXCLUSIONS row
    DROPPED entirely (not nulled) -- these are not steel observations and
    must not contribute to any steel cohort count, feature-coverage table,
    or model. The ORIGINAL df (and the underlying CSV it came from) is
    never modified -- callers needing the excluded rows for an audit report
    should filter the pre-exclusion dataframe with non_steel_scope_mask()
    instead of relying on this function's output. Idempotent: rows already
    absent simply don't match the mask again."""
    return df[~non_steel_scope_mask(df)].copy()


def apply_known_data_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a COPY of df with UTS_MPa_corrected (UTS_MPa with the one
    confirmed GPa->MPa conversion applied) and manual_unit_correction_applied
    added. Never mutates UTS_MPa itself -- the raw CSV value stays exactly
    as extracted; only the derived column is corrected. Must be called
    before target_cohort_mask() for UTS_MPa so the correction is visible to
    the primary/sensitivity split below. Also applies
    exclude_non_steel_scope() -- this is the single shared correction entry
    point every notebook calls, so a caller applying
    apply_known_data_corrections() automatically gets every known fix
    without needing to remember a second function. Non-steel rows are
    DROPPED (not nulled) -- see exclude_non_steel_scope's docstring for why
    a composition-only null was insufficient for these."""
    df = exclude_non_steel_scope(df)
    df["UTS_MPa_corrected"] = df["UTS_MPa"]
    df["manual_unit_correction_applied"] = False
    for (doi, obs_id), expected_raw in UTS_GPA_TO_MPA_CORRECTIONS.items():
        mask = (df["doi"] == doi) & (df["observation_id"] == obs_id)
        if not mask.any():
            continue
        actual = df.loc[mask, "UTS_MPa"].iloc[0]
        if actual != expected_raw:
            raise ValueError(
                f"UTS_GPA_TO_MPA_CORRECTIONS expected UTS_MPa=={expected_raw} for "
                f"({doi}, {obs_id}), found {actual} -- dataset changed since this "
                f"correction was audited; do not apply blindly."
            )
        df.loc[mask, "UTS_MPa_corrected"] = df.loc[mask, "UTS_MPa"] * 1000
        df.loc[mask, "manual_unit_correction_applied"] = True
    return df


TARGET_VALUE_COLUMNS = {"UTS_MPa": "UTS_MPa_corrected", "YS_MPa": "YS_MPa", "elongation_pct": "elongation_pct"}
TARGET_UNIT_VERIFIED_COLUMNS = {"UTS_MPa": "UTS_unit_verified", "YS_MPa": "YS_unit_verified", "elongation_pct": "elongation_unit_verified"}


def target_cohort_masks(df: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    """df MUST already have apply_known_data_corrections() applied (so
    UTS_MPa_corrected exists). Returns boolean Series (aligned to df.index)
    for three DISTINCT, non-overlapping-by-construction cohorts:
      primary    -- strict: unit-verified AND not any known exclusion.
      sensitivity -- lenient but never includes a CONFIRMED-wrong or
                     CANNOT-DECIDE value; DOES include CONFIRMED_INCONSISTENT_
                     PROBABLE_CAUSE rows' exclusion too (their inconsistency
                     IS confirmed, not merely probable -- see module docstring
                     above) -- so sensitivity is primary's superset only
                     insofar as it doesn't additionally require unit_verified.
      diagnostic_suspect_inclusive -- sensitivity PLUS the confirmed-
                     inconsistent rows put back in. Illustrates "what if no
                     QC had been done" -- NEVER report this as the robustness
                     result; it exists only to show examiners the effect of
                     skipping the audit above.
    """
    if target not in TARGET_VALUE_COLUMNS:
        raise ValueError(f"unknown target {target!r}")
    value_col = TARGET_VALUE_COLUMNS[target]
    unit_verified_col = TARGET_UNIT_VERIFIED_COLUMNS[target]

    def _key_mask(keys: set) -> pd.Series:
        if not keys:
            return pd.Series(False, index=df.index)
        pairs = set(keys)
        return df.apply(lambda r: (r["doi"], r["observation_id"]) in pairs, axis=1)

    has_target = df[value_col].notna()
    confirmed_wrong = _key_mask(CONFIRMED_WRONG_ELONGATION_KEYS) if target == "elongation_pct" else pd.Series(False, index=df.index)
    cannot_decide = _key_mask(CANNOT_DECIDE_KEYS.get(target, set()))
    confirmed_inconsistent = _key_mask(CONFIRMED_INCONSISTENT_PROBABLE_CAUSE_KEYS)

    sensitivity = has_target & ~confirmed_wrong & ~cannot_decide & ~confirmed_inconsistent
    # A manually-corrected row (currently: the one confirmed 1.2 GPa -> 1200 MPa case) has its
    # ORIGINAL unit_verified flag still False (that flag describes the raw extraction, not this
    # notebook's correction) -- it must count as verified for primary purposes once corrected.
    manually_corrected = df["manual_unit_correction_applied"] if "manual_unit_correction_applied" in df.columns else pd.Series(False, index=df.index)
    primary = sensitivity & ((df[unit_verified_col] == True) | manually_corrected)
    diagnostic_suspect_inclusive = sensitivity | (has_target & confirmed_inconsistent)

    return {"primary": primary, "sensitivity": sensitivity, "diagnostic_suspect_inclusive": diagnostic_suspect_inclusive}


# ══════════════════════════════════════════════════════════════════════════
# Configurable composition-qualifier resolver (no replicate resolver here —
# see module docstring)
# ══════════════════════════════════════════════════════════════════════════

def _default_qualifier_strategy(raw_value, qualifier):
    """Preserves any genuine numeric raw_value regardless of qualifier state.
    The default resolver makes NO scientific interpretation of qualifiers —
    it neither nulls an approximate numeric reading nor invents a number for
    a non-numeric entry (a bare 'trace'/'balance'/range-only string simply
    has no number to preserve, so pd.to_numeric coerces it to NaN on its
    own, not because of the qualifier). Any policy that changes a numeric
    value based on qualifier state (e.g. dropping 'trace' rows, converting
    a range to a midpoint) must be supplied explicitly as a custom
    qualifier_strategy — never the default."""
    return pd.to_numeric(raw_value, errors="coerce")


def resolve_composition_column(df: pd.DataFrame, element: str, qualifier_column: str | None = None,
                                qualifier_strategy: Callable | None = None) -> pd.Series:
    """Returns a numeric pandas Series for one composition element, using
    qualifier_strategy(raw_value, qualifier) -> float if given, else
    _default_qualifier_strategy (preserves every genuine numeric raw value,
    performs no qualifier-based interpretation). Qualifier metadata itself
    is left untouched in df[qualifier_column] — this function only ever
    returns the resolved numeric series, never mutates or drops the
    qualifier column. Pass a custom qualifier_strategy to opt into any
    qualifier-aware nulling/interpretation policy."""
    if element not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    strategy = qualifier_strategy or _default_qualifier_strategy
    raw = df[element]
    if qualifier_column and qualifier_column in df.columns:
        qual = df[qualifier_column]
    else:
        qual = pd.Series([None] * len(df), index=df.index)
    values = [strategy(r, q) for r, q in zip(raw, qual)]
    return pd.Series(values, index=df.index, dtype=float)


# ══════════════════════════════════════════════════════════════════════════
# Feature-list validation (must run before pipeline construction)
# ══════════════════════════════════════════════════════════════════════════

class FeatureListError(ValueError):
    pass


def validate_feature_lists(df: pd.DataFrame, numeric_features: Iterable[str], categorical_features: Iterable[str],
                            target: str | None = None, group_column: str = GROUP_COLUMN,
                            allow_target_or_group_as_feature: bool = False) -> None:
    """Raises FeatureListError (never silently ignores) if:
      - a requested feature column doesn't exist in df
      - a column is duplicated within a single feature list
      - a column appears in BOTH numeric_features and categorical_features
      - target or group_column is included as a feature, unless the caller
        explicitly opts in via allow_target_or_group_as_feature=True
    """
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    problems: list[str] = []

    missing = sorted({c for c in numeric_features + categorical_features if c not in df.columns})
    if missing:
        problems.append(f"feature columns not present in dataframe: {missing}")

    overlap = sorted(set(numeric_features) & set(categorical_features))
    if overlap:
        problems.append(f"columns listed in BOTH numeric_features and categorical_features: {overlap}")

    dup_numeric = sorted({c for c in numeric_features if numeric_features.count(c) > 1})
    if dup_numeric:
        problems.append(f"duplicate entries within numeric_features: {dup_numeric}")

    dup_categorical = sorted({c for c in categorical_features if categorical_features.count(c) > 1})
    if dup_categorical:
        problems.append(f"duplicate entries within categorical_features: {dup_categorical}")

    if not allow_target_or_group_as_feature:
        all_features = set(numeric_features) | set(categorical_features)
        if target is not None and target in all_features:
            problems.append(f"target column '{target}' is included in the feature lists "
                             f"— pass allow_target_or_group_as_feature=True if intentional")
        if group_column in all_features:
            problems.append(f"group column '{group_column}' is included in the feature lists "
                             f"— pass allow_target_or_group_as_feature=True if intentional")

    if problems:
        raise FeatureListError("; ".join(problems))


# ══════════════════════════════════════════════════════════════════════════
# Preprocessing + models
# ══════════════════════════════════════════════════════════════════════════

def build_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str],
                        scale_numeric: bool = True) -> ColumnTransformer:
    """Numeric imputation uses add_indicator=True (a missingness indicator
    column survives alongside the imputed value -- "this was measured" vs
    "this was imputed" stays visible to the model) and
    keep_empty_features=True (a feature that is ENTIRELY null within a given
    fold's training split is imputed to a constant rather than silently
    dropped by sklearn -- dropping would shift column alignment between
    folds and between train/test in a way that is easy to miss). A feature
    that actually goes empty in some fold is a real finding to report, not
    something to just tolerate quietly -- see fold_support_table below,
    which is meant to be checked BEFORE this is relied on for real, not
    treated as a silent safety net.

    scale_numeric=False skips StandardScaler for models where it makes no
    difference (tree ensembles) -- Ridge/SVR need scale_numeric=True."""
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                                      ("onehot", _make_onehot_encoder())])
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        raise FeatureListError("at least one of numeric_features/categorical_features must be non-empty")
    return ColumnTransformer(transformers)


# Models where StandardScaler measurably matters (distance/gradient-based) vs
# tree ensembles where it is a no-op on split quality -- explicit, not inferred.
MODELS_REQUIRING_SCALING = {"ridge", "svr"}


def fold_support_table(df: pd.DataFrame, feature_columns: Iterable[str], cv, groups,
                        group_column: str = GROUP_COLUMN) -> pd.DataFrame:
    """Per feature, across every training split cv.split() would produce:
    the MINIMUM non-null observation count and MINIMUM non-null DOI count
    seen in any training fold, and whether the feature was completely empty
    (zero non-null values) in at least one training fold. Must be checked
    BEFORE relying on keep_empty_features=True to quietly tolerate a
    completely-empty fold -- this table is what makes that situation visible
    rather than silently handled."""
    feature_columns = list(feature_columns)
    groups_arr = np.asarray(groups)
    n_nonnull_by_fold = {f: [] for f in feature_columns}
    n_doi_by_fold = {f: [] for f in feature_columns}
    dummy_y = np.zeros(len(df))
    for train_idx, _ in cv.split(df, dummy_y, groups_arr):
        train_df = df.iloc[train_idx]
        for f in feature_columns:
            present = train_df[f].notna()
            n_nonnull_by_fold[f].append(int(present.sum()))
            n_doi_by_fold[f].append(int(train_df.loc[present, group_column].nunique()))
    rows = []
    for f in feature_columns:
        rows.append({
            "feature": f,
            "min_train_n_observations": min(n_nonnull_by_fold[f]),
            "min_train_n_DOI": min(n_doi_by_fold[f]),
            "empty_in_any_training_fold": min(n_nonnull_by_fold[f]) == 0,
        })
    return pd.DataFrame(rows)


def _make_ridge(random_state: int = DEFAULT_RANDOM_STATE) -> BaseEstimator:
    return Ridge(alpha=1.0, random_state=random_state)


def _make_random_forest(random_state: int = DEFAULT_RANDOM_STATE) -> BaseEstimator:
    return RandomForestRegressor(n_estimators=100, random_state=random_state)


def _make_gradient_boosting(random_state: int = DEFAULT_RANDOM_STATE) -> BaseEstimator:
    return GradientBoostingRegressor(n_estimators=100, random_state=random_state)


def _make_svr(random_state: int = DEFAULT_RANDOM_STATE) -> BaseEstimator:
    return SVR()  # SVR has no random_state parameter; deterministic given fixed data/params


MODEL_FACTORIES: dict[str, Callable[..., BaseEstimator]] = {
    "ridge": _make_ridge,
    "random_forest": _make_random_forest,
    "gradient_boosting": _make_gradient_boosting,
    "svr": _make_svr,
}


def _make_dummy_mean(random_state: int = DEFAULT_RANDOM_STATE) -> BaseEstimator:
    return DummyRegressor(strategy="mean")  # random_state accepted for API uniformity, unused (no randomness)


# Kept OUT of MODEL_FACTORIES deliberately -- it is a reference line for
# interpreting the 4 research models' metrics (e.g. "does this model beat
# always-predicting-the-training-mean"), never one of the compared models
# itself, so list(MODEL_FACTORIES) stays exactly the 4-model research set.
BASELINE_MODEL_FACTORIES: dict[str, Callable[..., BaseEstimator]] = {"dummy_mean": _make_dummy_mean}
ALL_MODEL_FACTORIES: dict[str, Callable[..., BaseEstimator]] = {**MODEL_FACTORIES, **BASELINE_MODEL_FACTORIES}


def build_model_pipeline(model_name: str, numeric_features: Iterable[str], categorical_features: Iterable[str],
                          model_params: dict | None = None, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    if model_name not in ALL_MODEL_FACTORIES:
        raise ValueError(f"unknown model_name '{model_name}', available: {sorted(ALL_MODEL_FACTORIES)}")
    # Ridge/SVR are scale-sensitive; RF/GB split on raw feature values, so scaling is a
    # no-op for them -- explicit per-model policy, not a single blanket choice.
    preprocessor = build_preprocessor(numeric_features, categorical_features,
                                       scale_numeric=model_name in MODELS_REQUIRING_SCALING)
    model = ALL_MODEL_FACTORIES[model_name](random_state=random_state)
    if model_params:
        model.set_params(**model_params)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


# ══════════════════════════════════════════════════════════════════════════
# GroupKFold — fails clearly, never falls back to row-level CV
# ══════════════════════════════════════════════════════════════════════════

class InsufficientGroupsError(ValueError):
    pass


def make_group_kfold(df: pd.DataFrame, group_column: str = GROUP_COLUMN, min_splits: int = 3, max_splits: int = 10) -> GroupKFold:
    if group_column not in df.columns:
        raise InsufficientGroupsError(f"group column '{group_column}' not present in dataframe")
    n_groups = int(df[group_column].nunique(dropna=True))
    if n_groups < min_splits:
        raise InsufficientGroupsError(
            f"only {n_groups} unique group(s) in '{group_column}', need at least {min_splits} for "
            f"GroupKFold — refusing to silently fall back to row-level CV"
        )
    n_splits = max(min_splits, min(max_splits, n_groups))
    return GroupKFold(n_splits=n_splits)


# ══════════════════════════════════════════════════════════════════════════
# Metrics — R2 is NaN (not a crash, not a misleading value) when a fold has
# fewer than two target observations
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    """R2 is NaN — never a crash, never sklearn's own substitution — whenever
    it is statistically undefined: fewer than two observations, OR the
    held-out y_true has zero variance (a constant-target fold). sklearn's
    r2_score would otherwise silently return a value (commonly 0.0) for a
    constant target, which would misrepresent model performance in a
    thesis-facing metric. MAE/RMSE are still computed in both cases."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    if len(y_true) < 2 or np.std(y_true) == 0:
        r2 = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = r2_score(y_true, y_pred)
        r2 = float(r2) if np.isfinite(r2) else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


@dataclass
class CrossValidationResult:
    """fold_results: the existing per-fold train/test metric dicts (unchanged
    shape, consumed by aggregate_fold_metrics). oof_predictions: one row per
    held-out observation across all folds — the reusable basis for
    predicted-vs-actual plots, residual plots, bootstrap UQ, and paper-level
    error inspection in later notebooks. Columns: 'observation_id' (only
    present when X carries an 'observation_id' column — see
    cross_validate_model), 'index' (X's own index for that row), 'group'
    (the group/DOI value), 'fold', 'y_true', 'y_pred'."""
    fold_results: list[dict]
    oof_predictions: pd.DataFrame


def cross_validate_model(pipeline: Pipeline, X: pd.DataFrame, y, groups, cv) -> CrossValidationResult:
    """Runs the CV loop manually (not cross_val_score) so BOTH in-fold
    train metrics and held-out metrics are captured per fold — the
    structure the train-vs-CV comparison needs — and so the same held-out
    predict() call used for test_metrics is reused (not duplicated) to also
    populate oof_predictions.

    Optional observation_id traceability: if X has a column literally named
    'observation_id', it is carried through into oof_predictions unchanged
    (not required, not inferred from anything else). This is safe to pass
    through unconditionally — build_preprocessor's ColumnTransformer only
    ever touches the explicitly listed numeric_features/categorical_features
    columns and drops everything else, so an extra 'observation_id' column
    in X is never seen by, or used as input to, the model itself. Nothing
    about this parameter is mandatory: X without that column behaves exactly
    as before."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    has_observation_id = "observation_id" in X.columns
    fold_results = []
    oof_chunks = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        fold_pipeline = clone(pipeline)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fold_pipeline.fit(X_train, y_train)
        test_pred = fold_pipeline.predict(X_test)
        fold_results.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_metrics": compute_metrics(y_train, fold_pipeline.predict(X_train)),
            "test_metrics": compute_metrics(y_test, test_pred),
        })
        oof_row = {}
        if has_observation_id:
            oof_row["observation_id"] = X["observation_id"].values[test_idx]
        oof_row.update({
            "index": X.index[test_idx],
            "group": groups[test_idx],
            "fold": fold_idx,
            "y_true": y_test,
            "y_pred": test_pred,
        })
        oof_chunks.append(pd.DataFrame(oof_row))
    default_columns = (["observation_id"] if has_observation_id else []) + ["index", "group", "fold", "y_true", "y_pred"]
    oof_predictions = (pd.concat(oof_chunks, ignore_index=True) if oof_chunks
                        else pd.DataFrame(columns=default_columns))
    return CrossValidationResult(fold_results=fold_results, oof_predictions=oof_predictions)


def aggregate_fold_metrics(fold_results: list[dict]) -> pd.DataFrame:
    """Mean/std per metric, per train/cv split, across folds. NaN R2
    values (see compute_metrics) are excluded from the mean/std but
    counted in n_folds_finite so a caller can see how many folds actually
    contributed."""
    rows = []
    for split_key, split_name in (("train_metrics", "train"), ("test_metrics", "cv")):
        for metric in ("MAE", "RMSE", "R2"):
            values = np.array([fr[split_key][metric] for fr in fold_results], dtype=float)
            finite = values[np.isfinite(values)]
            rows.append({
                "split": split_name, "metric": metric,
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "std": float(np.std(finite)) if len(finite) else float("nan"),
                "n_folds": len(values), "n_folds_finite": int(len(finite)),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Model comparison (orchestration)
# ══════════════════════════════════════════════════════════════════════════

def run_target_experiment(df: pd.DataFrame, target: str, numeric_features: Iterable[str],
                           categorical_features: Iterable[str] = (), model_names: list[str] | None = None,
                           group_column: str = GROUP_COLUMN, min_splits: int = 3, max_splits: int = 10,
                           random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, CrossValidationResult]:
    """Runs every model in model_names on the SAME rows, SAME features, and
    the SAME GroupKFold split object (one cv is built once and reused --
    GroupKFold has no shuffling/random_state, so identical groups always
    produce identical splits; building it once, not once per model, is what
    actually guarantees this rather than merely making it likely). Returns
    {model_name: CrossValidationResult} so callers get BOTH the aggregate
    metrics (via aggregate_fold_metrics on each result) AND oof_predictions
    (for DOI-level diagnostics / plots) without running CV twice per model
    the way calling compare_models() and then cross_validate_model()
    separately would."""
    model_names = list(MODEL_FACTORIES) if model_names is None else list(model_names)
    validate_feature_lists(df, numeric_features, categorical_features, target=target, group_column=group_column)
    sub = df[df[target].notna()].copy()
    feature_cols = list(numeric_features) + list(categorical_features)
    passthrough_cols = ["observation_id"] if "observation_id" in df.columns else []
    X = sub[feature_cols + passthrough_cols]
    y = sub[target].values
    groups = sub[group_column].values
    cv = make_group_kfold(sub, group_column=group_column, min_splits=min_splits, max_splits=max_splits)

    results = {}
    for name in model_names:
        pipeline = build_model_pipeline(name, numeric_features, categorical_features, random_state=random_state)
        results[name] = cross_validate_model(pipeline, X, y, groups, cv)
    return results


def doi_level_mae_summary(oof_predictions: pd.DataFrame) -> dict:
    """Per-DOI mean absolute error, then the median and IQR ACROSS those
    per-DOI values (not the pooled per-observation MAE) -- shows whether a
    paper contributing many rows is dominating the pooled metric. Requires
    oof_predictions to have a 'group' column (always true for
    CrossValidationResult.oof_predictions)."""
    abs_error = (oof_predictions["y_true"] - oof_predictions["y_pred"]).abs()
    per_doi_mae = abs_error.groupby(oof_predictions["group"]).mean()
    return {
        "n_DOI": int(per_doi_mae.shape[0]),
        "median_DOI_MAE": float(per_doi_mae.median()),
        "IQR_DOI_MAE": float(per_doi_mae.quantile(0.75) - per_doi_mae.quantile(0.25)),
        "min_DOI_MAE": float(per_doi_mae.min()),
        "max_DOI_MAE": float(per_doi_mae.max()),
    }


def compare_models(df: pd.DataFrame, target: str, numeric_features: Iterable[str], categorical_features: Iterable[str],
                    model_names: list[str] | None = None, group_column: str = GROUP_COLUMN,
                    min_splits: int = 3, max_splits: int = 10, random_state: int = DEFAULT_RANDOM_STATE,
                    allow_target_or_group_as_feature: bool = False) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not present in dataframe")
    validate_feature_lists(df, numeric_features, categorical_features, target=target, group_column=group_column,
                            allow_target_or_group_as_feature=allow_target_or_group_as_feature)
    model_names = list(MODEL_FACTORIES) if model_names is None else list(model_names)

    sub = df[df[target].notna()].copy()
    feature_cols = list(numeric_features) + list(categorical_features)
    # Carry observation_id through to cross_validate_model's oof_predictions as metadata only,
    # when present -- never as a feature (it is not in feature_cols, so build_preprocessor's
    # ColumnTransformer never sees or uses it; see cross_validate_model's docstring).
    passthrough_cols = ["observation_id"] if "observation_id" in df.columns else []
    X = sub[feature_cols + passthrough_cols]
    y = sub[target].values
    cv = make_group_kfold(sub, group_column=group_column, min_splits=min_splits, max_splits=max_splits)
    groups = sub[group_column].values

    all_rows = []
    for name in model_names:
        pipeline = build_model_pipeline(name, numeric_features, categorical_features, random_state=random_state)
        cv_result = cross_validate_model(pipeline, X, y, groups, cv)
        agg = aggregate_fold_metrics(cv_result.fold_results)
        agg.insert(0, "model", name)
        agg.insert(0, "target", target)
        all_rows.append(agg)
    return pd.concat(all_rows, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════
# Target-specific quality tiers
# ══════════════════════════════════════════════════════════════════════════

class MissingProvenanceColumnError(ValueError):
    pass


def validate_provenance_columns_present(df: pd.DataFrame, provenance_map: ProvenanceColumnMap, target: str) -> None:
    """Raises MissingProvenanceColumnError if a column that was EXPLICITLY
    configured through ProvenanceColumnMap for this target's tiers is not
    actually present in df. This exists so a typo/schema mismatch in the
    caller-supplied column map produces a clear failure instead of silently
    collapsing every quality-tier predicate to all-False (which would look
    like "an empty high-confidence dataset" rather than "a broken config").
    Columns never referenced for this target (e.g. another target's
    provenance column, or the optional stage2_version_column) are not
    checked — only what's actually wired in for `target` here."""
    required_columns = {
        "material_identity_status_column": provenance_map.material_identity_status_column,
        "processing_identity_status_column": provenance_map.processing_identity_status_column,
    }
    if provenance_map.stage3_observation_status_column:
        required_columns["stage3_observation_status_column"] = provenance_map.stage3_observation_status_column

    target_provenance_col = provenance_map.target_provenance_verified_column.get(target)
    if target_provenance_col is None:
        raise MissingProvenanceColumnError(
            f"ProvenanceColumnMap.target_provenance_verified_column has no entry for target '{target}'"
        )
    required_columns[f"target_provenance_verified_column['{target}']"] = target_provenance_col

    missing = {role: col for role, col in required_columns.items() if col not in df.columns}
    if missing:
        raise MissingProvenanceColumnError(
            f"ProvenanceColumnMap references column(s) not present in the dataframe: {missing} "
            f"— refusing to silently build an all-False quality tier for target '{target}'"
        )


def build_target_quality_tiers(df: pd.DataFrame, target: str, provenance_map: ProvenanceColumnMap,
                                material_confirmed_values: Iterable, processing_confirmed_values: Iterable) -> list[QualityTierRule]:
    """Builds the three named tiers ('usable', 'validated', 'high_confidence')
    for exactly ONE target. A high-confidence tier for `target` checks ONLY
    that target's own provenance-verified flag — it never requires the
    other two targets to be provenance-verified, and missing YS/elongation
    never disqualifies an otherwise-usable UTS (or other target) row.

    Takes `df` (the dataframe these tiers will actually be applied to via
    apply_quality_tiers) so it can validate up front, via
    validate_provenance_columns_present, that every column this target's
    tiers reference actually exists — raising MissingProvenanceColumnError
    immediately rather than returning predicates that would silently
    evaluate to all-False against a typo'd/missing column."""
    validate_provenance_columns_present(df, provenance_map, target)
    usable_pred = make_target_has_value_predicate(target)

    validated_preds = [usable_pred]
    if provenance_map.stage3_observation_status_column:
        validated_preds.append(make_status_in_predicate(provenance_map.stage3_observation_status_column, {"valid_observation"}))
    validated_pred = make_and_predicate(*validated_preds)

    high_confidence_pred = make_and_predicate(
        validated_pred,
        make_status_in_predicate(provenance_map.material_identity_status_column, material_confirmed_values),
        make_status_in_predicate(provenance_map.processing_identity_status_column, processing_confirmed_values),
        make_target_provenance_predicate(provenance_map.target_provenance_verified_column, targets=[target]),
    )

    return [
        QualityTierRule(f"{target}_usable", usable_pred),
        QualityTierRule(f"{target}_validated", validated_pred),
        QualityTierRule(f"{target}_high_confidence", high_confidence_pred),
    ]


def apply_quality_tiers(df: pd.DataFrame, tiers: list[QualityTierRule]) -> dict[str, pd.DataFrame]:
    return {tier.name: df[tier.predicate(df)].copy() for tier in tiers}


def paper_matched_control(df_full: pd.DataFrame, df_tier: pd.DataFrame, group_column: str = GROUP_COLUMN) -> pd.DataFrame:
    """Restricts df_full to the DOIs (or other group_column values) present
    in df_tier. This matches which PAPERS are included — it does NOT match
    the number of observations, and must never be described as sample-size
    matching. Report paper count and observation count separately."""
    tier_groups = set(df_tier[group_column].unique())
    return df_full[df_full[group_column].isin(tier_groups)].copy()


# ══════════════════════════════════════════════════════════════════════════
# Bootstrap uncertainty (optional, for later) — defaults to group/cluster
# resampling, never silently falls back to row-level resampling
# ══════════════════════════════════════════════════════════════════════════

class BootstrapConfigError(ValueError):
    pass


def bootstrap_metric_ci(y_true, y_pred, groups=None, n_boot: int = 1000, seed: int = 42, row_level: bool = False) -> dict:
    """Default resampling is DOI/group-level cluster resampling: whole
    groups (all of a paper's (y_true, y_pred) pairs together) are resampled
    with replacement, not individual observations — observations from the
    same paper are not independent. `groups` is required for this default;
    a caller wanting plain row-level resampling must opt in explicitly with
    row_level=True rather than this function silently degrading."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)

    if not row_level and groups is None:
        raise BootstrapConfigError(
            "bootstrap_metric_ci defaults to group/cluster resampling and requires `groups` "
            "(e.g. the DOI for each observation); pass row_level=True explicitly to opt into "
            "plain row-level resampling instead"
        )

    maes, rmses, r2s = [], [], []
    if row_level:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m = compute_metrics(y_true[idx], y_pred[idx])
            maes.append(m["MAE"]); rmses.append(m["RMSE"]); r2s.append(m["R2"])
        resampling = "row_level"
    else:
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        n_groups = len(unique_groups)
        group_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}
        for _ in range(n_boot):
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_indices[g] for g in sampled_groups])
            m = compute_metrics(y_true[idx], y_pred[idx])
            maes.append(m["MAE"]); rmses.append(m["RMSE"]); r2s.append(m["R2"])
        resampling = "group_level"

    def _summarize(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite):
            ci_lower, ci_upper = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
        else:
            ci_lower, ci_upper = float("nan"), float("nan")
        return {"mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "std": float(np.std(finite)) if len(finite) else float("nan"),
                "ci_lower": ci_lower, "ci_upper": ci_upper}

    return {"MAE": _summarize(maes), "RMSE": _summarize(rmses), "R2": _summarize(r2s),
            "n_boot": n_boot, "resampling": resampling}


def bootstrap_metric_contrast(y_true, y_pred_a, y_pred_b, groups, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired DOI-cluster bootstrap contrast between two prediction arrays evaluated on the
    SAME observations (e.g. a model vs. the dummy-mean reference on identical OOF rows).
    Always resamples whole DOI groups with replacement (never row-level -- observations from
    the same paper are not independent); in every replicate, delta_MAE = MAE(a) - MAE(b) and
    delta_RMSE = RMSE(a) - RMSE(b) are computed on the SAME resampled rows for both `a` and
    `b`, so this is a genuinely paired contrast, not two independent bootstraps subtracted
    afterward. Negative values mean `a` outperforms `b`. Returns the point estimate (on the
    full, unresampled data) plus a 95% percentile interval for each delta -- never a p-value
    or a significant/not-significant label; callers must not convert this into one."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    groups_arr = np.asarray(groups)
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b) == len(groups_arr)):
        raise ValueError("y_true, y_pred_a, y_pred_b, and groups must all be the same length")

    point_a = compute_metrics(y_true, y_pred_a)
    point_b = compute_metrics(y_true, y_pred_b)
    point_delta_mae = point_a["MAE"] - point_b["MAE"]
    point_delta_rmse = point_a["RMSE"] - point_b["RMSE"]

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups_arr)
    n_groups = len(unique_groups)
    group_indices = {g: np.where(groups_arr == g)[0] for g in unique_groups}

    delta_maes, delta_rmses = [], []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        m_a = compute_metrics(y_true[idx], y_pred_a[idx])
        m_b = compute_metrics(y_true[idx], y_pred_b[idx])
        delta_maes.append(m_a["MAE"] - m_b["MAE"])
        delta_rmses.append(m_a["RMSE"] - m_b["RMSE"])

    def _interval(point: float, values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite):
            lo, hi = float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))
        else:
            lo, hi = float("nan"), float("nan")
        return {"point_estimate": float(point), "ci_lower": lo, "ci_upper": hi}

    return {
        "delta_MAE": _interval(point_delta_mae, delta_maes),
        "delta_RMSE": _interval(point_delta_rmse, delta_rmses),
        "n_boot": n_boot, "n_obs": int(len(y_true)), "n_DOI": int(n_groups),
    }


# ══════════════════════════════════════════════════════════════════════════
# CONTEXTUAL-COMPLETENESS ERROR DIAGNOSTICS — Notebook 04/05. Fixed OOF
# predictions in, per-subgroup descriptive error statistics out. Never
# computes a subgroup R², t-test, or p-value: association-only, descriptive.
# ══════════════════════════════════════════════════════════════════════════

def support_label(n_doi: int, exploratory_min_doi: int = 5, supported_min_doi: int = 10) -> str:
    """Names the evidentiary weight of a subgroup by its unique-DOI count only
    (never by n_observations, which can be inflated by one paper reporting many
    rows). Labels are descriptive support tiers, NOT statistical significance."""
    if n_doi >= supported_min_doi:
        return "supported_descriptive"
    if n_doi >= exploratory_min_doi:
        return "exploratory"
    return "insufficient_support"


def error_subgroup_table(oof: pd.DataFrame, group_col: str, group_column: str = "doi",
                          exploratory_min_doi: int = 5, supported_min_doi: int = 10) -> pd.DataFrame:
    """Per-group descriptive error table from fixed OOF predictions: n observations, n unique
    DOI, MAE, median absolute error, support label. `oof` must have 'y_true', 'y_pred',
    `group_column` (DOI), and `group_col` (the subgroup label, e.g. processing status).
    Deliberately never computes a subgroup R² (unstable/misleading at small n) or any
    hypothesis test -- descriptive statistics only."""
    work = oof.copy()
    work["_abs_error"] = (work["y_true"] - work["y_pred"]).abs()
    rows = []
    for group_value, g in work.groupby(group_col, dropna=False, observed=True):
        n_doi = int(g[group_column].nunique())
        rows.append({
            group_col: group_value,
            "n_obs": int(len(g)),
            "n_DOI": n_doi,
            "MAE": float(g["_abs_error"].mean()),
            "median_abs_error": float(g["_abs_error"].median()),
            "support": support_label(n_doi, exploratory_min_doi, supported_min_doi),
        })
    return pd.DataFrame(rows)


def tie_safe_quantile_groups(values: pd.Series, n_groups: int = 3, labels: list[str] | None = None) -> dict:
    """Quantile-based grouping (e.g. low/medium/high) that automatically REDUCES the number of
    groups when the requested quantile boundaries collide (heavily tied values), rather than
    splitting identical values arbitrarily across group boundaries. Tries n_groups, then
    n_groups-1, ... down to 1, using pandas' own duplicate-edge handling
    (`pd.qcut(..., duplicates='drop')`) and stopping at the largest count that actually
    produces that many distinct groups. Returns the group labels (aligned to `values`' index),
    the actual bin edges used, and per-group sizes, so a caller reports what was actually
    formed instead of assuming n_groups groups exist."""
    default_labels = ["low", "medium", "high"]
    values = pd.Series(values)
    binned, edges, k_used = None, None, 0
    for k in range(n_groups, 0, -1):
        candidate_labels = (labels[:k] if labels is not None
                             else (default_labels[:k] if k <= 3 else [f"q{i + 1}" for i in range(k)]))
        try:
            candidate_binned, candidate_edges = pd.qcut(values, k, labels=candidate_labels,
                                                          retbins=True, duplicates="drop")
        except (ValueError, IndexError):
            continue
        if int(candidate_binned.nunique(dropna=True)) == k:
            binned, edges, k_used = candidate_binned, candidate_edges, k
            break
    if binned is None:
        # Every value is identical -- one degenerate group, not an error.
        single_label = (labels[0] if labels else default_labels[0])
        binned = pd.Series(single_label, index=values.index)
        binned[values.isna()] = np.nan
        edges = [float(values.min()), float(values.max())] if values.notna().any() else [float("nan"), float("nan")]
        k_used = 1
    return {
        "groups": binned,
        "n_groups_formed": k_used,
        "boundaries": [float(e) for e in edges],
        "group_sizes": binned.value_counts(dropna=True).to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════
# TARGET FEATURE CONFIGURATION — decided in 02_feature_preparation.ipynb from
# real per-target composition coverage (observation- AND DOI-level). Lives
# here, not as a notebook-local variable, so 02 and 03 both call the same
# function on a freshly-loaded+corrected dataframe rather than 03 importing
# a variable out of 02's executed session (which would make the two
# notebooks NOT independent -- 03 would silently depend on 02 having been
# run first in the same kernel).
# ══════════════════════════════════════════════════════════════════════════

def composition_coverage_table(df: pd.DataFrame, primary_mask: pd.Series,
                                composition_elements: Iterable[str] = None) -> pd.DataFrame:
    """Per-element observation-level AND DOI-level coverage within the given
    primary-cohort mask. Both are reported because an element present in
    many ROWS that all come from one PAPER is a materially different
    situation from one spread across many papers -- row coverage alone
    cannot distinguish them."""
    from literature_pipeline.modelling.dataset_contract import CORE_COMPOSITION_ELEMENTS, GROUP_COLUMN
    elements = list(composition_elements) if composition_elements is not None else CORE_COMPOSITION_ELEMENTS
    primary_df = df[primary_mask]
    n_obs = len(primary_df)
    n_doi = primary_df[GROUP_COLUMN].nunique()
    rows = []
    for element in elements:
        present = primary_df[element].notna()
        rows.append({
            "element": element,
            "n_observations": int(present.sum()),
            "observation_coverage": round(present.sum() / n_obs, 3) if n_obs else float("nan"),
            "n_unique_DOI": primary_df.loc[present, GROUP_COLUMN].nunique(),
            "DOI_coverage": round(primary_df.loc[present, GROUP_COLUMN].nunique() / n_doi, 3) if n_doi else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("observation_coverage", ascending=False).reset_index(drop=True)


DEFAULT_MIN_OBSERVATIONS_FOR_FEATURE = 20
DEFAULT_MIN_DOI_FOR_FEATURE = 5


def build_target_configs(df: pd.DataFrame, min_observations: int = DEFAULT_MIN_OBSERVATIONS_FOR_FEATURE,
                          min_doi: int = DEFAULT_MIN_DOI_FOR_FEATURE) -> dict:
    """The single, shared source of TARGET_CONFIGS -- df must already have
    apply_known_data_corrections() applied. Selection rule (fixed BEFORE any
    model is fit, identical across 02/03): a composition element is retained
    for a given target if it is present in at least min_observations
    observations AND at least min_doi independent DOI within that target's
    own primary cohort. processing_key is never included (see
    02_feature_preparation.ipynb's investigation: 22/26 values are
    single-DOI, a DOI-fingerprinting risk, not a physical feature)."""
    from literature_pipeline.modelling.dataset_contract import TARGET_COLUMNS, GROUP_COLUMN
    configs = {}
    for target in TARGET_COLUMNS:
        masks = target_cohort_masks(df, target)
        coverage = composition_coverage_table(df, masks["primary"])
        selected = coverage.loc[
            (coverage["n_observations"] >= min_observations) & (coverage["n_unique_DOI"] >= min_doi)
        ]
        configs[target] = {
            "display_target": target,  # human-facing label, e.g. in plot titles/print statements
            "model_target_column": TARGET_VALUE_COLUMNS[target],  # the ACTUAL column fit/predicted -- for UTS_MPa
            # this is "UTS_MPa_corrected", never raw "UTS_MPa", so the 1.2->1200 GPa fix is used, not silently bypassed.
            "value_column": TARGET_VALUE_COLUMNS[target],  # kept for backward compatibility with existing callers
            "cohort": "primary",
            "selection_rule": f"n_observations >= {min_observations} AND n_unique_DOI >= {min_doi} "
                               f"(within this target's own primary cohort)",
            "coverage_table": coverage,
            "feature_sets": {"composition": selected["element"].tolist()},
            "categorical_features": [],
            "composition_qualifier_strategy": None,
            "exclude_zero_composition_rows": True,
            "primary_n": int(masks["primary"].sum()),
            "primary_DOI": df.loc[masks["primary"], GROUP_COLUMN].nunique(),
        }
    return configs
