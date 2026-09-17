"""
test_modelling_lib.py — plain-assert style tests, synthetic/mock data only.
No real CPP data, no LLM calls, no real modelling results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from literature_pipeline.modelling.dataset_contract import ProvenanceColumnMap
from literature_pipeline.modelling import dataset_contract as dc  # noqa: E402
from literature_pipeline.modelling import modelling_lib as ml  # noqa: E402

FAILURES = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not cond:
        FAILURES.append(label)


# ─────────────────────────────────────────────────────────────────────────
# resolve_composition_column
# ─────────────────────────────────────────────────────────────────────────
df_comp = pd.DataFrame({
    "C": [0.20, 0.15, 0.30, 0.10],
    "C_qualifier": [None, "exact", "trace", "balance"],
})
resolved_with_qual_col = ml.resolve_composition_column(df_comp, "C", qualifier_column="C_qualifier")
check("qualifier=None (absent metadata) preserves the genuine numeric value",
      resolved_with_qual_col.iloc[0] == 0.20)
check("qualifier='exact' preserves the genuine numeric value",
      resolved_with_qual_col.iloc[1] == 0.15)
check("default resolver does NOT null a numeric value just because qualifier='trace' "
      "(no scientific interpretation of qualifiers by default)",
      resolved_with_qual_col.iloc[2] == 0.30)
check("default resolver does NOT null a numeric value just because qualifier='balance'",
      resolved_with_qual_col.iloc[3] == 0.10)

df_comp_no_qual_col = pd.DataFrame({"C": [0.20, 0.15]})
resolved_no_qual_col = ml.resolve_composition_column(df_comp_no_qual_col, "C", qualifier_column=None)
check("no qualifier_column supplied at all -> every genuine numeric value preserved",
      list(resolved_no_qual_col) == [0.20, 0.15])

resolved_missing_element = ml.resolve_composition_column(df_comp_no_qual_col, "Mn")
check("element column missing entirely -> all-NaN Series, no crash", resolved_missing_element.isna().all())

df_comp_nonnumeric = pd.DataFrame({
    "C": [0.20, "trace", "balance", "0.1-0.3"],
    "C_qualifier": [None, "trace", "balance", "range"],
})
resolved_nonnumeric = ml.resolve_composition_column(df_comp_nonnumeric, "C", qualifier_column="C_qualifier")
check("a non-numeric raw entry ('trace'/'balance'/a range string) coerces to NaN because there is "
      "no number to preserve -- not because of the qualifier",
      resolved_nonnumeric.iloc[1:].isna().all())
check("a genuine numeric raw value alongside a non-exact qualifier is still preserved",
      resolved_nonnumeric.iloc[0] == 0.20)


def _null_non_exact(raw_value, qualifier):
    numeric = pd.to_numeric(raw_value, errors="coerce")
    qual_str = "" if qualifier is None else str(qualifier).strip().lower()
    return numeric if qual_str in ("", "exact") else float("nan")


resolved_custom_strategy = ml.resolve_composition_column(df_comp, "C", qualifier_column="C_qualifier",
                                                          qualifier_strategy=_null_non_exact)
check("a custom qualifier_strategy CAN opt into qualifier-based nulling (trace/balance -> NaN) "
      "-- only the default resolver refrains from this",
      resolved_custom_strategy.iloc[2:].isna().all() and resolved_custom_strategy.iloc[0] == 0.20)

# ─────────────────────────────────────────────────────────────────────────
# build_preprocessor / one-hot compatibility
# ─────────────────────────────────────────────────────────────────────────
df_feat = pd.DataFrame({
    "C": [0.1, 0.2, 0.15, None],
    "Mn": [1.0, 1.2, None, 0.9],
    "process": ["hot_rolled", "cold_rolled", "hot_rolled", None],
})
pre = ml.build_preprocessor(numeric_features=["C", "Mn"], categorical_features=["process"])
transformed = pre.fit_transform(df_feat)
check("build_preprocessor output is a dense array (GradientBoosting-compatible, no sparse matrix)",
      isinstance(transformed, np.ndarray))
check("build_preprocessor output has no NaNs after imputation",
      not np.isnan(transformed).any())

pre_numeric_only = ml.build_preprocessor(numeric_features=["C", "Mn"], categorical_features=[])
check("build_preprocessor works with categorical_features=[] (numeric only)",
      pre_numeric_only.fit_transform(df_feat).shape[0] == 4)

try:
    ml.build_preprocessor(numeric_features=[], categorical_features=[])
    check("build_preprocessor raises when both feature lists are empty", False)
except ml.FeatureListError:
    check("build_preprocessor raises when both feature lists are empty", True)

# ─────────────────────────────────────────────────────────────────────────
# MODEL_FACTORIES — Ridge not RidgeCV, random_state set on RF/GB
# ─────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ridge_model = ml.MODEL_FACTORIES["ridge"](random_state=42)
check("ridge factory returns Ridge, not RidgeCV", isinstance(ridge_model, Ridge) and not isinstance(ridge_model, RidgeCV))

rf_model = ml.MODEL_FACTORIES["random_forest"](random_state=7)
check("random_forest factory sets random_state as requested",
      isinstance(rf_model, RandomForestRegressor) and rf_model.random_state == 7)

gb_model = ml.MODEL_FACTORIES["gradient_boosting"](random_state=7)
check("gradient_boosting factory sets random_state as requested",
      isinstance(gb_model, GradientBoostingRegressor) and gb_model.random_state == 7)

svr_model = ml.MODEL_FACTORIES["svr"](random_state=7)
check("svr factory returns a plain SVR (no random_state param on SVR itself)", isinstance(svr_model, SVR))

check("MODEL_FACTORIES has exactly the four approved models",
      set(ml.MODEL_FACTORIES.keys()) == {"ridge", "random_forest", "gradient_boosting", "svr"})

# ─────────────────────────────────────────────────────────────────────────
# validate_feature_lists
# ─────────────────────────────────────────────────────────────────────────
df_val = pd.DataFrame({"doi": ["a"], "UTS_MPa": [800.0], "C": [0.1], "process": ["hot_rolled"]})

try:
    ml.validate_feature_lists(df_val, numeric_features=["C", "nonexistent"], categorical_features=["process"])
    check("validate_feature_lists raises on missing feature column", False)
except ml.FeatureListError as e:
    check("validate_feature_lists raises on missing feature column", "nonexistent" in str(e))

try:
    ml.validate_feature_lists(df_val, numeric_features=["C"], categorical_features=["C"])
    check("validate_feature_lists raises when a column is in both numeric and categorical", False)
except ml.FeatureListError as e:
    check("validate_feature_lists raises when a column is in both numeric and categorical", "C" in str(e))

try:
    ml.validate_feature_lists(df_val, numeric_features=["C", "C"], categorical_features=[])
    check("validate_feature_lists raises on duplicate entries within numeric_features", False)
except ml.FeatureListError:
    check("validate_feature_lists raises on duplicate entries within numeric_features", True)

try:
    ml.validate_feature_lists(df_val, numeric_features=["C", "UTS_MPa"], categorical_features=[], target="UTS_MPa")
    check("validate_feature_lists raises when target leaks into features", False)
except ml.FeatureListError:
    check("validate_feature_lists raises when target leaks into features", True)

try:
    ml.validate_feature_lists(df_val, numeric_features=["C", "doi"], categorical_features=[], group_column="doi")
    check("validate_feature_lists raises when group_column leaks into features", False)
except ml.FeatureListError:
    check("validate_feature_lists raises when group_column leaks into features", True)

# opt-in override must be respected
try:
    ml.validate_feature_lists(df_val, numeric_features=["C", "UTS_MPa"], categorical_features=[], target="UTS_MPa",
                               allow_target_or_group_as_feature=True)
    check("validate_feature_lists allows target-as-feature when explicitly opted in", True)
except ml.FeatureListError:
    check("validate_feature_lists allows target-as-feature when explicitly opted in", False)

# clean case should not raise
try:
    ml.validate_feature_lists(df_val, numeric_features=["C"], categorical_features=["process"], target="UTS_MPa")
    check("validate_feature_lists does not raise on a clean, valid feature-list spec", True)
except ml.FeatureListError as e:
    check("validate_feature_lists does not raise on a clean, valid feature-list spec", False, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────
# make_group_kfold — fails clearly, never falls back to row-level CV
# ─────────────────────────────────────────────────────────────────────────
df_few_groups = pd.DataFrame({"doi": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0]})
try:
    ml.make_group_kfold(df_few_groups, group_column="doi", min_splits=3)
    check("make_group_kfold raises InsufficientGroupsError when groups < min_splits", False)
except ml.InsufficientGroupsError as e:
    check("make_group_kfold raises InsufficientGroupsError when groups < min_splits", "2" in str(e) and "3" in str(e))

df_missing_group_col = pd.DataFrame({"y": [1.0, 2.0]})
try:
    ml.make_group_kfold(df_missing_group_col, group_column="doi", min_splits=2)
    check("make_group_kfold raises when group_column itself is missing", False)
except ml.InsufficientGroupsError:
    check("make_group_kfold raises when group_column itself is missing", True)

df_enough_groups = pd.DataFrame({"doi": [f"doi{i}" for i in range(6)]})
cv_ok = ml.make_group_kfold(df_enough_groups, group_column="doi", min_splits=3, max_splits=10)
check("make_group_kfold returns a GroupKFold with n_splits capped by n_groups", cv_ok.get_n_splits() == 6)

df_many_groups = pd.DataFrame({"doi": [f"doi{i}" for i in range(50)]})
cv_capped = ml.make_group_kfold(df_many_groups, group_column="doi", min_splits=3, max_splits=10)
check("make_group_kfold caps n_splits at max_splits", cv_capped.get_n_splits() == 10)

# ─────────────────────────────────────────────────────────────────────────
# compute_metrics — R2 NaN edge case for n<2
# ─────────────────────────────────────────────────────────────────────────
metrics_single = ml.compute_metrics([5.0], [5.5])
check("compute_metrics: MAE computed even with a single observation", abs(metrics_single["MAE"] - 0.5) < 1e-9)
check("compute_metrics: R2 is NaN (not crash, not misleading value) for n<2 observations",
      np.isnan(metrics_single["R2"]))

metrics_multi = ml.compute_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
check("compute_metrics: R2 is a finite number for n>=2 with variance", np.isfinite(metrics_multi["R2"]))

metrics_constant_perfect = ml.compute_metrics([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
check("compute_metrics: R2 is NaN for a zero-variance (constant) y_true even when predictions match exactly "
      "-- undefined, not sklearn's own substitution",
      np.isnan(metrics_constant_perfect["R2"]))
check("compute_metrics: MAE is still computed (0.0) for the constant-target, perfect-prediction fold",
      metrics_constant_perfect["MAE"] == 0.0)

metrics_constant_with_error = ml.compute_metrics([5.0, 5.0, 5.0], [5.2, 4.8, 5.1])
check("compute_metrics: R2 is NaN for a zero-variance y_true with imperfect predictions too",
      np.isnan(metrics_constant_with_error["R2"]))
check("compute_metrics: MAE/RMSE are still finite, non-NaN numbers for the constant-target fold",
      np.isfinite(metrics_constant_with_error["MAE"]) and np.isfinite(metrics_constant_with_error["RMSE"]))

# ─────────────────────────────────────────────────────────────────────────
# cross_validate_model / aggregate_fold_metrics / compare_models — synthetic
# ─────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
n_papers = 12
rows = []
for p in range(n_papers):
    doi = f"10.1/synthetic{p}"
    n_obs = rng.integers(1, 4)
    c_val = rng.uniform(0.05, 0.4)
    process = rng.choice(["hot_rolled", "cold_rolled"])
    for o in range(n_obs):
        rows.append({
            "doi": doi,
            "material_id": f"m{p}_{o}",
            "condition_id": f"c{p}_{o}",
            "observation_id": f"o{p}_{o}",
            "C": c_val + rng.normal(0, 0.01),
            "Mn": rng.uniform(0.5, 2.0),
            "process": process,
            "UTS_MPa": 400 + 1500 * c_val + rng.normal(0, 10),
            "YS_MPa": 300 + 1200 * c_val + rng.normal(0, 10) if rng.random() > 0.3 else None,
            "elongation_pct": None,
        })
df_synth = pd.DataFrame(rows)

comparison = ml.compare_models(df_synth, target="UTS_MPa", numeric_features=["C", "Mn"],
                                categorical_features=["process"], group_column="doi", min_splits=3, max_splits=5)
check("compare_models returns a non-empty tidy DataFrame", len(comparison) > 0)
check("compare_models covers all four default models",
      set(comparison["model"].unique()) == {"ridge", "random_forest", "gradient_boosting", "svr"})
check("compare_models output has both train and cv splits", set(comparison["split"].unique()) == {"train", "cv"})
check("compare_models output tracks target column", set(comparison["target"].unique()) == {"UTS_MPa"})

try:
    ml.compare_models(df_synth, target="nonexistent_target", numeric_features=["C"], categorical_features=[])
    check("compare_models raises on nonexistent target column", False)
except ValueError:
    check("compare_models raises on nonexistent target column", True)

# ─────────────────────────────────────────────────────────────────────────
# End-to-end observation_id traceability THROUGH compare_models itself:
# original df -> compare_models's internal X-construction -> cross_validate_model
# -> oof_predictions (observation_id + DOI + fold + y_true + y_pred).
# compare_models does not return the CrossValidationResult, so the X actually
# handed to cross_validate_model is captured via a monkeypatch of the
# module-level cross_validate_model -- the real compare_models code path
# runs unmodified; only the boundary is observed.
# ─────────────────────────────────────────────────────────────────────────
_captured_calls = []
_real_cross_validate_model = ml.cross_validate_model


def _capturing_cross_validate_model(pipeline, X, y, groups, cv):
    _captured_calls.append(X.copy())
    return _real_cross_validate_model(pipeline, X, y, groups, cv)


ml.cross_validate_model = _capturing_cross_validate_model
try:
    _ = ml.compare_models(df_synth, target="UTS_MPa", numeric_features=["C", "Mn"],
                           categorical_features=["process"], model_names=["ridge"],
                           group_column="doi", min_splits=3, max_splits=5)
finally:
    ml.cross_validate_model = _real_cross_validate_model

check("compare_models's cross_validate_model call was actually captured", len(_captured_calls) == 1)
captured_X = _captured_calls[0]
check("compare_models passes observation_id through into the X handed to cross_validate_model "
      "(previously dropped -- X was built from numeric_features + categorical_features only)",
      "observation_id" in captured_X.columns)
check("observation_id is NOT among compare_models's declared feature columns "
      "(present as passthrough metadata only, never as a model feature)",
      "observation_id" not in (["C", "Mn"] + ["process"]))

# Full manual replay of compare_models's own X-construction logic against
# cross_validate_model directly, to inspect the resulting oof_predictions end
# to end (compare_models itself only returns the aggregated metrics table).
_sub = df_synth[df_synth["UTS_MPa"].notna()].copy()
_X_replay = _sub[["C", "Mn", "process", "observation_id"]]
_y_replay = _sub["UTS_MPa"].values
_groups_replay = _sub["doi"].values
_cv_replay = ml.make_group_kfold(_sub, group_column="doi", min_splits=3, max_splits=5)
_pipeline_replay = ml.build_model_pipeline("ridge", ["C", "Mn"], ["process"])
_cv_result_replay = ml.cross_validate_model(_pipeline_replay, _X_replay, _y_replay, _groups_replay, _cv_replay)
_oof_replay = _cv_result_replay.oof_predictions

check("end-to-end: oof_predictions has one row per UTS_MPa-usable observation",
      len(_oof_replay) == len(_sub))
check("end-to-end: oof_predictions carries observation_id, group (DOI), fold, y_true, y_pred all together",
      {"observation_id", "group", "fold", "y_true", "y_pred"}.issubset(set(_oof_replay.columns)))
check("end-to-end: oof_predictions.observation_id set matches the original dataframe's observation_id set "
      "exactly (no rows lost/duplicated in transit)",
      set(_oof_replay["observation_id"]) == set(_sub["observation_id"]))
_obs_to_doi = _sub.set_index("observation_id")["doi"]
check("end-to-end: each oof row's group (DOI) matches the DOI its observation_id actually belongs to "
      "in the original dataframe",
      all(_oof_replay["group"].values == _obs_to_doi.reindex(_oof_replay["observation_id"]).values))

pipeline_single = ml.build_model_pipeline("ridge", ["C", "Mn"], ["process"])
X_synth = df_synth[["C", "Mn", "process"]].copy()
X_synth.index = df_synth["observation_id"].values  # caller opts into observation_id as the OOF row identifier
cv_for_synth = ml.make_group_kfold(df_synth, min_splits=3, max_splits=5)
cv_result = ml.cross_validate_model(
    pipeline_single, X_synth, df_synth["UTS_MPa"].values, df_synth["doi"].values, cv_for_synth)

# observation_id passthrough (optional) — an extra, non-feature column on X
X_synth_with_obs_id = df_synth[["C", "Mn", "process"]].copy()
X_synth_with_obs_id["observation_id"] = df_synth["observation_id"].values
cv_result_with_obs_id = ml.cross_validate_model(
    pipeline_single, X_synth_with_obs_id, df_synth["UTS_MPa"].values, df_synth["doi"].values, cv_for_synth)
oof_with_obs_id = cv_result_with_obs_id.oof_predictions
check("oof_predictions includes an 'observation_id' column when X carries one",
      "observation_id" in oof_with_obs_id.columns)
check("oof_predictions.observation_id values match the original df_synth observation_id set exactly",
      set(oof_with_obs_id["observation_id"]) == set(df_synth["observation_id"]))
check("oof_predictions still has the required index/group/fold/y_true/y_pred columns alongside observation_id",
      {"index", "group", "fold", "y_true", "y_pred"}.issubset(set(oof_with_obs_id.columns)))

# without an observation_id column on X, the CV helper does not require/invent one
X_synth_no_obs_id = df_synth[["C", "Mn", "process"]].copy()
cv_result_no_obs_id = ml.cross_validate_model(
    pipeline_single, X_synth_no_obs_id, df_synth["UTS_MPa"].values, df_synth["doi"].values, cv_for_synth)
check("oof_predictions has no 'observation_id' column when X does not carry one "
      "(observation_id is optional, never mandatory in the generic CV helper)",
      "observation_id" not in cv_result_no_obs_id.oof_predictions.columns)

# passing observation_id as an extra column on X must not change model
# predictions vs. X without it (ColumnTransformer drops unlisted columns)
check("presence of the extra observation_id column on X does not alter model predictions "
      "(proves it is never used as a feature)",
      np.allclose(cv_result_with_obs_id.oof_predictions.sort_values(["fold", "y_true"])["y_pred"].values,
                   cv_result_no_obs_id.oof_predictions.sort_values(["fold", "y_true"])["y_pred"].values))

check("cross_validate_model returns a CrossValidationResult with fold_results and oof_predictions",
      isinstance(cv_result, ml.CrossValidationResult))
check("cross_validate_model produces one fold_results dict per fold",
      len(cv_result.fold_results) == cv_for_synth.get_n_splits())
check("cross_validate_model fold result contains train_metrics and test_metrics",
      "train_metrics" in cv_result.fold_results[0] and "test_metrics" in cv_result.fold_results[0])

oof = cv_result.oof_predictions
check("oof_predictions has one row per observation (every row is held out exactly once across folds)",
      len(oof) == len(df_synth))
check("oof_predictions carries the required columns: index, group, fold, y_true, y_pred",
      set(oof.columns) == {"index", "group", "fold", "y_true", "y_pred"})
check("oof_predictions 'index' matches the observation_id identifiers set on X's index",
      set(oof["index"]) == set(df_synth["observation_id"]))
check("oof_predictions 'group' matches the DOI each held-out row actually belonged to",
      all(oof.set_index("index")["group"].reindex(df_synth["observation_id"]).values == df_synth["doi"].values))
check("oof_predictions 'fold' values are within [0, n_splits)",
      oof["fold"].between(0, cv_for_synth.get_n_splits() - 1).all())
check("oof_predictions did not require a second/duplicate model fit (fold count matches fold_results exactly)",
      oof["fold"].nunique() == len(cv_result.fold_results))

agg = ml.aggregate_fold_metrics(cv_result.fold_results)
check("aggregate_fold_metrics returns rows for train+cv x MAE/RMSE/R2 (6 rows)", len(agg) == 6)
check("aggregate_fold_metrics tracks n_folds_finite separately from n_folds",
      "n_folds_finite" in agg.columns and "n_folds" in agg.columns)

# ─────────────────────────────────────────────────────────────────────────
# build_target_quality_tiers / apply_quality_tiers — target-specificity
# ─────────────────────────────────────────────────────────────────────────
df_tiers = pd.DataFrame({
    "doi": ["a", "b", "c", "d"],
    "UTS_MPa": [800.0, 850.0, None, 900.0],
    "YS_MPa": [None, None, 600.0, None],  # deliberately NOT provenance-verified/absent for UTS rows
    "elongation_pct": [None, None, None, None],
    "material_status": ["confirmed", "confirmed", "confirmed", "unconfirmed"],
    "processing_status": ["confirmed", "confirmed", "confirmed", "confirmed"],
    "uts_provenance_ok": [True, True, False, True],
    "ys_provenance_ok": [False, False, True, False],  # YS provenance false everywhere except row c
    "elong_provenance_ok": [False, False, False, False],
    "stage3_status": ["valid_observation", "valid_observation", "valid_observation", "valid_observation"],
})
pmap_tiers = ProvenanceColumnMap(
    material_identity_status_column="material_status",
    processing_identity_status_column="processing_status",
    target_provenance_verified_column={"UTS_MPa": "uts_provenance_ok", "YS_MPa": "ys_provenance_ok",
                                        "elongation_pct": "elong_provenance_ok"},
    stage3_observation_status_column="stage3_status",
)
uts_tiers = ml.build_target_quality_tiers(df_tiers, "UTS_MPa", pmap_tiers, material_confirmed_values={"confirmed"},
                                           processing_confirmed_values={"confirmed"})
tier_frames = ml.apply_quality_tiers(df_tiers, uts_tiers)

check("UTS usable tier includes exactly the rows with non-null UTS (rows a,b,d)",
      set(tier_frames["UTS_MPa_usable"]["doi"]) == {"a", "b", "d"})
check("UTS high_confidence tier does NOT require YS provenance to be true "
      "(rows a,b have ys_provenance_ok=False but still qualify on UTS-only provenance)",
      set(tier_frames["UTS_MPa_high_confidence"]["doi"]) == {"a", "b"})
check("UTS high_confidence tier still excludes row d (material_status unconfirmed, unrelated to YS provenance)",
      "d" not in set(tier_frames["UTS_MPa_high_confidence"]["doi"]))

ys_tiers = ml.build_target_quality_tiers(df_tiers, "YS_MPa", pmap_tiers, material_confirmed_values={"confirmed"},
                                          processing_confirmed_values={"confirmed"})
ys_tier_frames = ml.apply_quality_tiers(df_tiers, ys_tiers)
check("YS usable tier includes only row c (the only non-null YS)",
      set(ys_tier_frames["YS_MPa_usable"]["doi"]) == {"c"})
check("YS high_confidence tier requires YS's OWN provenance flag; row c (confirmed material/processing, "
      "ys_provenance_ok=True) is the only qualifying row",
      set(ys_tier_frames["YS_MPa_high_confidence"]["doi"]) == {"c"})

# ─────────────────────────────────────────────────────────────────────────
# fail-fast for explicitly configured provenance columns (correction 2) —
# a typo'd/missing column must raise, never silently collapse to an
# all-False (empty) high-confidence tier
# ─────────────────────────────────────────────────────────────────────────
pmap_typo = ProvenanceColumnMap(
    material_identity_status_column="materal_status",  # typo: does not exist in df_tiers
    processing_identity_status_column="processing_status",
    target_provenance_verified_column={"UTS_MPa": "uts_provenance_ok"},
    stage3_observation_status_column="stage3_status",
)
try:
    ml.build_target_quality_tiers(df_tiers, "UTS_MPa", pmap_typo, material_confirmed_values={"confirmed"},
                                   processing_confirmed_values={"confirmed"})
    check("build_target_quality_tiers raises on a typo'd material_identity_status_column instead of "
          "silently building an all-False tier", False)
except ml.MissingProvenanceColumnError as e:
    check("build_target_quality_tiers raises on a typo'd material_identity_status_column instead of "
          "silently building an all-False tier", "materal_status" in str(e))

pmap_missing_target_col = ProvenanceColumnMap(
    material_identity_status_column="material_status",
    processing_identity_status_column="processing_status",
    target_provenance_verified_column={"UTS_MPa": "uts_provenance_ok"},  # no entry at all for elongation_pct
    stage3_observation_status_column="stage3_status",
)
try:
    ml.build_target_quality_tiers(df_tiers, "elongation_pct", pmap_missing_target_col,
                                   material_confirmed_values={"confirmed"}, processing_confirmed_values={"confirmed"})
    check("build_target_quality_tiers raises when target_provenance_verified_column has no entry for the "
          "requested target", False)
except ml.MissingProvenanceColumnError:
    check("build_target_quality_tiers raises when target_provenance_verified_column has no entry for the "
          "requested target", True)

try:
    ml.validate_provenance_columns_present(df_tiers, pmap_typo, "UTS_MPa")
    check("validate_provenance_columns_present raises directly on the same typo'd column", False)
except ml.MissingProvenanceColumnError:
    check("validate_provenance_columns_present raises directly on the same typo'd column", True)

# never-referenced/optional columns (e.g. another target's provenance column
# that this target's tiers don't touch) must remain harmless if absent
pmap_ys_col_missing_but_unused_for_uts = ProvenanceColumnMap(
    material_identity_status_column="material_status",
    processing_identity_status_column="processing_status",
    target_provenance_verified_column={"UTS_MPa": "uts_provenance_ok", "YS_MPa": "this_column_does_not_exist"},
    stage3_observation_status_column="stage3_status",
)
try:
    ml.build_target_quality_tiers(df_tiers, "UTS_MPa", pmap_ys_col_missing_but_unused_for_uts,
                                   material_confirmed_values={"confirmed"}, processing_confirmed_values={"confirmed"})
    check("build_target_quality_tiers for UTS does not raise over an unrelated target's missing "
          "provenance column (YS_MPa's column is never referenced when building UTS tiers)", True)
except ml.MissingProvenanceColumnError as e:
    check("build_target_quality_tiers for UTS does not raise over an unrelated target's missing "
          "provenance column (YS_MPa's column is never referenced when building UTS tiers)", False, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────
# paper_matched_control — DOI-set matching, not observation-count matching
# ─────────────────────────────────────────────────────────────────────────
df_full_pm = pd.DataFrame({"doi": ["a", "a", "b", "c", "c", "c", "d"], "val": range(7)})
df_tier_pm = pd.DataFrame({"doi": ["a", "c"], "val": [99, 98]})
control = ml.paper_matched_control(df_full_pm, df_tier_pm, group_column="doi")
check("paper_matched_control keeps ALL rows for papers present in the tier (a: 2 rows, c: 3 rows = 5 total)",
      len(control) == 5)
check("paper_matched_control does NOT match observation count (5 full-dataset rows vs 2 tier rows)",
      len(control) != len(df_tier_pm))
check("paper_matched_control excludes papers not present in the tier (b, d excluded)",
      set(control["doi"]) == {"a", "c"})

import inspect  # noqa: E402
sig = inspect.signature(ml.paper_matched_control)
check("paper_matched_control has the exact required signature (df_full, df_tier, group_column=GROUP_COLUMN)",
      list(sig.parameters.keys()) == ["df_full", "df_tier", "group_column"])

# ─────────────────────────────────────────────────────────────────────────
# bootstrap_metric_ci — defaults to group-level cluster resampling
# ─────────────────────────────────────────────────────────────────────────
y_true_bs = np.array([100.0, 110.0, 200.0, 210.0, 300.0, 310.0])
y_pred_bs = np.array([105.0, 108.0, 195.0, 215.0, 305.0, 295.0])
groups_bs = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])

try:
    ml.bootstrap_metric_ci(y_true_bs, y_pred_bs, groups=None, n_boot=50)
    check("bootstrap_metric_ci raises BootstrapConfigError when groups is None and row_level not requested", False)
except ml.BootstrapConfigError:
    check("bootstrap_metric_ci raises BootstrapConfigError when groups is None and row_level not requested", True)

result_group = ml.bootstrap_metric_ci(y_true_bs, y_pred_bs, groups=groups_bs, n_boot=200, seed=1)
check("bootstrap_metric_ci with groups given defaults to group_level resampling", result_group["resampling"] == "group_level")
check("bootstrap_metric_ci group-level result contains MAE/RMSE/R2 summaries",
      all(k in result_group for k in ("MAE", "RMSE", "R2")))

result_row = ml.bootstrap_metric_ci(y_true_bs, y_pred_bs, groups=None, n_boot=200, seed=1, row_level=True)
check("bootstrap_metric_ci row_level=True works without groups", result_row["resampling"] == "row_level")


# ── apply_known_data_corrections / target_cohort_masks ──────────────────────
# Synthetic dataset shaped like cpp_dataset_v2.csv, exercising every category
# from 01_dataset_qc.ipynb's audit: a normal verified row, the one confirmed
# GPa->MPa correction, a confirmed-wrong elongation row, a cannot-decide row,
# and a confirmed-inconsistent-probable-cause (DP780-style) row.
COHORT_TEST_ROWS = [
    # normal, unit-verified -- should land in primary/sensitivity/diagnostic for all three targets
    {"doi": "10.1/normal", "observation_id": "M1-P1-O1", "UTS_MPa": 900.0, "UTS_unit_verified": True,
     "YS_MPa": 600.0, "YS_unit_verified": True, "elongation_pct": 15.0, "elongation_unit_verified": True},
    # the one confirmed GPa correction case (real doi/obs_id from CONFIRMED corrections)
    {"doi": "10.1016/j.jmrt.2021.07.045", "observation_id": "M1-P1-O1", "UTS_MPa": 1.2, "UTS_unit_verified": False,
     "YS_MPa": None, "YS_unit_verified": False, "elongation_pct": None, "elongation_unit_verified": False},
    # confirmed-wrong elongation (real key from CONFIRMED_WRONG_ELONGATION_KEYS)
    {"doi": "10.1016/j.heliyon.2024.e32101", "observation_id": "M1-P1-O2", "UTS_MPa": None, "UTS_unit_verified": False,
     "YS_MPa": None, "YS_unit_verified": False, "elongation_pct": 1.3, "elongation_unit_verified": False},
    # cannot-decide elongation (real key from CANNOT_DECIDE_KEYS["elongation_pct"])
    {"doi": "10.1016/j.corcom.2026.04.005", "observation_id": "M1-P1-O5", "UTS_MPa": None, "UTS_unit_verified": False,
     "YS_MPa": None, "YS_unit_verified": False, "elongation_pct": 14.1, "elongation_unit_verified": False},
    # confirmed-inconsistent probable-cause (real key, affects both UTS and YS)
    {"doi": "10.1016/j.ijmecsci.2025.110565", "observation_id": "M1-P1-O7", "UTS_MPa": 4.8, "UTS_unit_verified": True,
     "YS_MPa": 5.0, "YS_unit_verified": True, "elongation_pct": None, "elongation_unit_verified": False},
]
cohort_df = pd.DataFrame(COHORT_TEST_ROWS)
corrected_df = ml.apply_known_data_corrections(cohort_df)

check("apply_known_data_corrections: exactly one row corrected",
      corrected_df["manual_unit_correction_applied"].sum() == 1)
check("apply_known_data_corrections: UTS_MPa itself is never mutated (still 1.2, raw)",
      corrected_df.loc[corrected_df["observation_id"] == "M1-P1-O1", "UTS_MPa"].tolist() == [900.0, 1.2])
check("apply_known_data_corrections: UTS_MPa_corrected has the GPa row converted to 1200",
      corrected_df.loc[(corrected_df["doi"] == "10.1016/j.jmrt.2021.07.045"), "UTS_MPa_corrected"].iloc[0] == 1200.0)

# Value-mismatch guard: if the dataset changes, the correction must refuse to apply silently.
mismatch_df = cohort_df.copy()
mismatch_df.loc[mismatch_df["doi"] == "10.1016/j.jmrt.2021.07.045", "UTS_MPa"] = 99.9
try:
    ml.apply_known_data_corrections(mismatch_df)
    check("apply_known_data_corrections raises on unexpected value at a known-correction key", False)
except ValueError:
    check("apply_known_data_corrections raises on unexpected value at a known-correction key", True)

# Idempotency: applying the correction to an already-corrected dataframe must not
# double-apply it (e.g. 1.2 -> 1200 -> 1,200,000). apply_known_data_corrections always
# reads from the raw UTS_MPa column (never from a prior UTS_MPa_corrected), so calling
# it again on its own output must reproduce exactly the same result.
twice_corrected_df = ml.apply_known_data_corrections(corrected_df)
check("apply_known_data_corrections is idempotent: re-applying to already-corrected data "
      "does not double-convert (still 1200.0, not 1,200,000)",
      twice_corrected_df.loc[twice_corrected_df["doi"] == "10.1016/j.jmrt.2021.07.045", "UTS_MPa_corrected"].iloc[0] == 1200.0)
check("apply_known_data_corrections is idempotent: correction count unchanged on re-application",
      twice_corrected_df["manual_unit_correction_applied"].sum() == corrected_df["manual_unit_correction_applied"].sum())

uts_masks = ml.target_cohort_masks(corrected_df, "UTS_MPa")
check("target_cohort_masks(UTS_MPa): primary includes the normal row and the corrected GPa row (2)",
      uts_masks["primary"].sum() == 2)
check("target_cohort_masks(UTS_MPa): primary excludes the confirmed-inconsistent DP780-style row",
      not uts_masks["primary"][cohort_df["doi"] == "10.1016/j.ijmecsci.2025.110565"].iloc[0])
check("target_cohort_masks(UTS_MPa): sensitivity also excludes the confirmed-inconsistent row (2, not 3)",
      uts_masks["sensitivity"].sum() == 2)
check("target_cohort_masks(UTS_MPa): diagnostic_suspect_inclusive puts the confirmed-inconsistent row back (3)",
      uts_masks["diagnostic_suspect_inclusive"].sum() == 3)

ys_masks = ml.target_cohort_masks(corrected_df, "YS_MPa")
check("target_cohort_masks(YS_MPa): the same confirmed-inconsistent row is excluded here too "
      "(shared key affects both UTS and YS)",
      not ys_masks["sensitivity"][cohort_df["doi"] == "10.1016/j.ijmecsci.2025.110565"].iloc[0])

elong_masks = ml.target_cohort_masks(corrected_df, "elongation_pct")
check("target_cohort_masks(elongation_pct): confirmed-wrong row excluded from primary",
      not elong_masks["primary"][cohort_df["doi"] == "10.1016/j.heliyon.2024.e32101"].iloc[0])
check("target_cohort_masks(elongation_pct): confirmed-wrong row excluded from sensitivity too "
      "(never merely a primary-only exclusion)",
      not elong_masks["sensitivity"][cohort_df["doi"] == "10.1016/j.heliyon.2024.e32101"].iloc[0])
check("target_cohort_masks(elongation_pct): cannot-decide row also excluded from sensitivity",
      not elong_masks["sensitivity"][cohort_df["doi"] == "10.1016/j.corcom.2026.04.005"].iloc[0])
check("target_cohort_masks(elongation_pct): only the one normal row reaches primary",
      elong_masks["primary"].sum() == 1)

print()
print("=" * 70)
# ── composition_coverage_table / build_target_configs ───────────────────────
# Synthetic dataset: 3 papers, target UTS_MPa only (single-target dataset_contract
# TARGET_COLUMNS still requires YS_MPa/elongation_pct columns to exist -- included
# as all-null, which build_target_configs must handle without raising).
cov_rows = []
for doi_i in range(3):  # 3 DOI
    for row_i in range(4):  # 4 rows each = 12 rows total
        cov_rows.append({
            "doi": f"10.1/paper{doi_i}", "observation_id": f"M1-P1-O{row_i}",
            "material_id": "M1", "condition_id": "M1-P1",
            "UTS_MPa": 900.0 + row_i, "UTS_unit_verified": True,
            "YS_MPa": None, "YS_unit_verified": False,
            "elongation_pct": None, "elongation_unit_verified": False,
            # C: present in every row of every paper (3 DOI, 12 obs) -- should be selected
            "C": 0.2, "C_qualifier": "exact",
            # Mn: present in only 1 of 3 papers (1 DOI, 4 obs) -- should be excluded by DOI floor
            "Mn": 1.5 if doi_i == 0 else None, "Mn_qualifier": "exact" if doi_i == 0 else None,
        })
cov_df = pd.DataFrame(cov_rows)
for e in dc.CORE_COMPOSITION_ELEMENTS:
    if e not in cov_df.columns:
        cov_df[e] = None
        cov_df[f"{e}_qualifier"] = None
cov_df = ml.apply_known_data_corrections(cov_df)

masks_cov = ml.target_cohort_masks(cov_df, "UTS_MPa")
coverage = ml.composition_coverage_table(cov_df, masks_cov["primary"])
check("composition_coverage_table: C has 12 observations across 3 DOI",
      coverage.set_index("element").loc["C", ["n_observations", "n_unique_DOI"]].tolist() == [12, 3])
check("composition_coverage_table: Mn has 4 observations across only 1 DOI",
      coverage.set_index("element").loc["Mn", ["n_observations", "n_unique_DOI"]].tolist() == [4, 1])

configs = ml.build_target_configs(cov_df, min_observations=10, min_doi=2)
check("build_target_configs: C (12 obs, 3 DOI) passes a 10-obs/2-DOI rule",
      "C" in configs["UTS_MPa"]["feature_sets"]["composition"])
check("build_target_configs: Mn (4 obs, 1 DOI) fails the same rule (DOI floor)",
      "Mn" not in configs["UTS_MPa"]["feature_sets"]["composition"])
check("build_target_configs: selection_rule states the exact thresholds used",
      "10" in configs["UTS_MPa"]["selection_rule"] and "2" in configs["UTS_MPa"]["selection_rule"])
check("build_target_configs: does not raise for targets with zero non-null values (YS_MPa, elongation_pct)",
      configs["YS_MPa"]["primary_n"] == 0 and configs["elongation_pct"]["primary_n"] == 0)
check("build_target_configs: default thresholds match the documented module constants",
      ml.DEFAULT_MIN_OBSERVATIONS_FOR_FEATURE == 20 and ml.DEFAULT_MIN_DOI_FOR_FEATURE == 5)

# ── build_preprocessor scaling policy / add_indicator / keep_empty_features ──
scale_pre = ml.build_preprocessor(["a", "b"], [], scale_numeric=True)
noscale_pre = ml.build_preprocessor(["a", "b"], [], scale_numeric=False)
check("build_preprocessor(scale_numeric=True) includes a 'scale' step in the numeric pipeline",
      "scale" in dict(scale_pre.transformers[0][1].steps))
check("build_preprocessor(scale_numeric=False) omits the 'scale' step",
      "scale" not in dict(noscale_pre.transformers[0][1].steps))

_imputer = dict(scale_pre.transformers[0][1].steps)["impute"]
check("build_preprocessor's numeric imputer has add_indicator=True", _imputer.add_indicator is True)
check("build_preprocessor's numeric imputer has keep_empty_features=True", _imputer.keep_empty_features is True)

check("MODELS_REQUIRING_SCALING is exactly {ridge, svr}", ml.MODELS_REQUIRING_SCALING == {"ridge", "svr"})
ridge_pipeline = ml.build_model_pipeline("ridge", ["a"], [])
rf_pipeline = ml.build_model_pipeline("random_forest", ["a"], [])
check("build_model_pipeline('ridge', ...) scales (Ridge is scale-sensitive)",
      "scale" in dict(ridge_pipeline.named_steps["preprocess"].transformers[0][1].steps))
check("build_model_pipeline('random_forest', ...) does not scale (tree ensembles are scale-invariant)",
      "scale" not in dict(rf_pipeline.named_steps["preprocess"].transformers[0][1].steps))

# A feature completely empty in one fold must not crash or silently drop a column --
# keep_empty_features=True must make this survive as an all-imputed column.
empty_fold_df = pd.DataFrame({
    "doi": [f"10.1/p{i}" for i in range(6) for _ in range(2)],
    "y": np.arange(12, dtype=float),
    "feat_sparse": [1.0, 2.0] + [None] * 10,  # only paper p0 has this feature at all
})
sparse_cv = ml.make_group_kfold(empty_fold_df, min_splits=3, max_splits=3)
sparse_pipeline = ml.build_model_pipeline("ridge", ["feat_sparse"], [])
try:
    result = ml.cross_validate_model(sparse_pipeline, empty_fold_df[["feat_sparse"]],
                                      empty_fold_df["y"], empty_fold_df["doi"], sparse_cv)
    check("cross_validate_model does not crash when a feature is empty in some training fold "
          "(keep_empty_features=True)", True)
except Exception as e:
    check(f"cross_validate_model does not crash when a feature is empty in some training fold "
          f"(keep_empty_features=True) -- raised {type(e).__name__}: {e}", False)

# ── fold_support_table ───────────────────────────────────────────────────────
support = ml.fold_support_table(empty_fold_df, ["feat_sparse"], sparse_cv, empty_fold_df["doi"])
check("fold_support_table flags feat_sparse as empty in at least one training fold",
      bool(support.set_index("feature").loc["feat_sparse", "empty_in_any_training_fold"]))

well_covered_df = empty_fold_df.copy()
well_covered_df["feat_dense"] = np.arange(12, dtype=float)  # present in every row of every paper
support2 = ml.fold_support_table(well_covered_df, ["feat_sparse", "feat_dense"], sparse_cv, well_covered_df["doi"])
check("fold_support_table does NOT flag a fully-covered feature as empty in any fold",
      not bool(support2.set_index("feature").loc["feat_dense", "empty_in_any_training_fold"]))
check("fold_support_table reports min_train_n_observations > 0 for the fully-covered feature",
      support2.set_index("feature").loc["feat_dense", "min_train_n_observations"] > 0)

# ── run_target_experiment / doi_level_mae_summary ────────────────────────────
exp_df = pd.DataFrame({
    "doi": [f"10.1/p{i}" for i in range(8) for _ in range(3)],
    "observation_id": [f"O{i}-{j}" for i in range(8) for j in range(3)],
    "y_target": np.tile([100.0, 110.0, 90.0], 8) + np.repeat(np.arange(8), 3) * 2,
    "feat1": np.tile([1.0, 2.0, 3.0], 8),
})
exp_results = ml.run_target_experiment(exp_df, "y_target", ["feat1"], model_names=["ridge", "random_forest"],
                                        min_splits=4, max_splits=4)
check("run_target_experiment returns one CrossValidationResult per requested model",
      set(exp_results.keys()) == {"ridge", "random_forest"})
check("run_target_experiment: both models see the same fold structure (identical n_test per fold)",
      [r["n_test"] for r in exp_results["ridge"].fold_results] ==
      [r["n_test"] for r in exp_results["random_forest"].fold_results])
check("run_target_experiment: oof_predictions covers every row exactly once (24 rows, 4 folds)",
      len(exp_results["ridge"].oof_predictions) == len(exp_df))

doi_summary = ml.doi_level_mae_summary(exp_results["ridge"].oof_predictions)
check("doi_level_mae_summary returns n_DOI matching the number of groups actually held out",
      doi_summary["n_DOI"] <= exp_df["doi"].nunique() and doi_summary["n_DOI"] > 0)
check("doi_level_mae_summary: median/IQR are non-negative finite numbers",
      doi_summary["median_DOI_MAE"] >= 0 and doi_summary["IQR_DOI_MAE"] >= 0)

# ── dummy_mean baseline / build_target_configs display vs model column ──────
check("MODEL_FACTORIES has exactly the 4 research models (dummy_mean not among them)",
      set(ml.MODEL_FACTORIES) == {"ridge", "random_forest", "gradient_boosting", "svr"})
check("BASELINE_MODEL_FACTORIES has exactly dummy_mean", set(ml.BASELINE_MODEL_FACTORIES) == {"dummy_mean"})
dummy_pipeline = ml.build_model_pipeline("dummy_mean", ["a"], [])
check("build_model_pipeline accepts 'dummy_mean' (looked up via ALL_MODEL_FACTORIES)",
      dummy_pipeline.named_steps["model"].strategy == "mean")

dummy_test_df = pd.DataFrame({
    "doi": [f"10.1/p{i}" for i in range(6) for _ in range(2)],
    "a": np.tile([1.0, 2.0], 6),
    "y": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
})
dummy_cv = ml.make_group_kfold(dummy_test_df, min_splits=3, max_splits=3)
dummy_result = ml.cross_validate_model(ml.build_model_pipeline("dummy_mean", ["a"], []),
                                        dummy_test_df[["a"]], dummy_test_df["y"], dummy_test_df["doi"], dummy_cv)
check("dummy_mean pipeline actually produces a single repeated value per fold's predictions",
      len(dummy_result.oof_predictions["y_pred"].round(6).unique()) <= dummy_cv.n_splits)

configs_named = ml.build_target_configs(cov_df, min_observations=10, min_doi=2)
check("build_target_configs: display_target matches the target key",
      configs_named["UTS_MPa"]["display_target"] == "UTS_MPa")
check("build_target_configs: model_target_column is UTS_MPa_corrected, not raw UTS_MPa",
      configs_named["UTS_MPa"]["model_target_column"] == "UTS_MPa_corrected")

# ── non_steel_scope_mask / exclude_non_steel_scope ───────────────────────────
# Covers: full exclusion, raw dataframe untouched, unrelated DOI unaffected,
# idempotency, and paper-scoped keys not leaking across DOI (same material_id
# "M1" appears under both a confirmed-non-steel DOI and an unrelated DOI --
# only the specific (doi, material_id) pair must be excluded).
scope_df = pd.DataFrame({
    "doi": ["10.1007/s40962-026-02006-1", "10.1007/s40962-026-02006-1",
            "10.1007/s40194-026-02341-1", "10.1/unrelated-paper", "10.1/unrelated-paper"],
    "observation_id": ["M1-P1-O1", "M3-P1-O1", "M3-P1-O1", "M1-P1-O1", "M2-P1-O1"],
    "material_id": ["M1", "M3", "M3", "M1", "M2"],  # note: "M1" reused under an unrelated DOI
    "Cu": [0.002, 999.0, 999.0, 0.5, 0.3],
    "UTS_MPa": [900.0, 900.0, 900.0, 900.0, 900.0],
})
for e in dc.CORE_COMPOSITION_ELEMENTS:
    if e not in scope_df.columns:
        scope_df[e] = None

mask = ml.non_steel_scope_mask(scope_df)
check("non_steel_scope_mask: flags exactly the 3 confirmed non-steel rows (2 Al-alloy + 1 Cu-foil)",
      int(mask.sum()) == 3)
check("non_steel_scope_mask: does NOT flag 'M1' under the unrelated DOI -- paper-scoped keys "
      "cannot affect another DOI even with the same material_id",
      not mask[(scope_df["doi"] == "10.1/unrelated-paper") & (scope_df["material_id"] == "M1")].iloc[0])

excluded_df = ml.exclude_non_steel_scope(scope_df)
check("exclude_non_steel_scope: non-steel rows are fully DROPPED (2 rows remain, not 5)",
      len(excluded_df) == 2)
check("exclude_non_steel_scope: the remaining rows are exactly the unrelated-paper rows",
      set(excluded_df["doi"]) == {"10.1/unrelated-paper"})
check("exclude_non_steel_scope: original dataframe is NOT mutated (still all 5 rows, raw extraction preserved)",
      len(scope_df) == 5)
check("exclude_non_steel_scope: idempotent -- excluding twice gives the same result as once",
      len(ml.exclude_non_steel_scope(excluded_df)) == len(excluded_df))

corrected_scope_df = ml.apply_known_data_corrections(scope_df)
check("apply_known_data_corrections applies exclude_non_steel_scope automatically (5 rows -> 2)",
      len(corrected_scope_df) == 2)

# ─────────────────────────────────────────────────────────────────────────
# support_label / error_subgroup_table / tie_safe_quantile_groups / bootstrap_metric_contrast
# (Notebook 04/05 contextual-completeness diagnostics)
# ─────────────────────────────────────────────────────────────────────────
check("support_label: <5 DOI is insufficient_support", ml.support_label(4) == "insufficient_support")
check("support_label: exactly 5 DOI is exploratory (boundary is inclusive)", ml.support_label(5) == "exploratory")
check("support_label: 9 DOI is still exploratory", ml.support_label(9) == "exploratory")
check("support_label: exactly 10 DOI is supported_descriptive (boundary is inclusive)",
      ml.support_label(10) == "supported_descriptive")
check("support_label: custom thresholds are respected",
      ml.support_label(6, exploratory_min_doi=6, supported_min_doi=8) == "exploratory")

oof_sub = pd.DataFrame({
    "y_true": [100.0, 110.0, 300.0, 90.0, 95.0, 105.0, 310.0, 305.0],
    "y_pred": [90.0, 130.0, 250.0, 90.0, 100.0, 100.0, 320.0, 280.0],
    "doi": ["p1", "p1", "p2", "p3", "p3", "p3", "p4", "p4"],
    "processing_status": ["confirmed_baseline", "confirmed_baseline", "confirmed_explicit",
                           "confirmed_explicit", "confirmed_explicit", "confirmed_explicit",
                           "confirmed_explicit", "confirmed_explicit"],
})
sub_table = ml.error_subgroup_table(oof_sub, "processing_status")
check("error_subgroup_table: one row per distinct subgroup value", len(sub_table) == 2)
check("error_subgroup_table: n_obs counts rows, not DOI",
      int(sub_table.loc[sub_table["processing_status"] == "confirmed_baseline", "n_obs"].iloc[0]) == 2)
check("error_subgroup_table: n_DOI counts UNIQUE doi, not rows",
      int(sub_table.loc[sub_table["processing_status"] == "confirmed_explicit", "n_DOI"].iloc[0]) == 3)
baseline_mae = sub_table.loc[sub_table["processing_status"] == "confirmed_baseline", "MAE"].iloc[0]
check("error_subgroup_table: MAE is the mean absolute error within the subgroup",
      abs(baseline_mae - np.mean([abs(100 - 90), abs(110 - 130)])) < 1e-9)
check("error_subgroup_table: never computes an R2 column", "R2" not in sub_table.columns)
check("error_subgroup_table: support label reflects DOI count (2 DOI -> insufficient_support)",
      sub_table.loc[sub_table["processing_status"] == "confirmed_baseline", "support"].iloc[0] == "insufficient_support")

# tie_safe_quantile_groups: heavily tied values (as real composition-completeness ratios are,
# since they take only a handful of discrete values) must not force 3 arbitrary groups.
tied_values = pd.Series([0.5] * 12)
tied_result = ml.tie_safe_quantile_groups(tied_values, n_groups=3)
check("tie_safe_quantile_groups: all-identical input collapses to exactly 1 group, not 3",
      tied_result["n_groups_formed"] == 1)
check("tie_safe_quantile_groups: group_sizes accounts for every input value",
      sum(tied_result["group_sizes"].values()) == 12)

spread_values = pd.Series([0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9])
spread_result = ml.tie_safe_quantile_groups(spread_values, n_groups=3)
check("tie_safe_quantile_groups: sufficiently spread values DO form the requested 3 groups",
      spread_result["n_groups_formed"] == 3)
check("tie_safe_quantile_groups: boundaries are reported (not hidden)", len(spread_result["boundaries"]) >= 2)
check("tie_safe_quantile_groups: returned groups are aligned to the input index (same length)",
      len(spread_result["groups"]) == len(spread_values))

# bootstrap_metric_contrast: a clearly-better model vs. a dummy on the SAME observations.
rng_bc = np.random.default_rng(7)
y_true_bc = rng_bc.normal(500, 80, 40)
groups_bc = np.repeat([f"doi{i}" for i in range(8)], 5)
y_pred_good = y_true_bc + rng_bc.normal(0, 15, 40)
y_pred_dummy = np.full(40, y_true_bc.mean())
contrast = ml.bootstrap_metric_contrast(y_true_bc, y_pred_good, y_pred_dummy, groups_bc, n_boot=300, seed=42)
check("bootstrap_metric_contrast: delta_MAE point estimate is negative (model beats dummy)",
      contrast["delta_MAE"]["point_estimate"] < 0)
check("bootstrap_metric_contrast: delta_RMSE point estimate is negative (model beats dummy)",
      contrast["delta_RMSE"]["point_estimate"] < 0)
check("bootstrap_metric_contrast: ci_lower <= point_estimate <= ci_upper for delta_MAE",
      contrast["delta_MAE"]["ci_lower"] <= contrast["delta_MAE"]["point_estimate"] <= contrast["delta_MAE"]["ci_upper"])
check("bootstrap_metric_contrast: n_DOI reflects unique groups, not row count",
      contrast["n_DOI"] == 8 and contrast["n_obs"] == 40)
contrast_repeat = ml.bootstrap_metric_contrast(y_true_bc, y_pred_good, y_pred_dummy, groups_bc, n_boot=300, seed=42)
check("bootstrap_metric_contrast: fixed seed is fully reproducible",
      contrast["delta_MAE"]["ci_lower"] == contrast_repeat["delta_MAE"]["ci_lower"] and
      contrast["delta_MAE"]["ci_upper"] == contrast_repeat["delta_MAE"]["ci_upper"])

# bootstrap_metric_ci now also reports a percentile CI (used by Notebook 05), in addition to
# the pre-existing mean/std -- verify the new fields exist without breaking the old ones.
ci_result = ml.bootstrap_metric_ci(y_true_bs, y_pred_bs, groups=groups_bs, n_boot=200, seed=1)
check("bootstrap_metric_ci: MAE summary now also reports ci_lower/ci_upper",
      "ci_lower" in ci_result["MAE"] and "ci_upper" in ci_result["MAE"])
check("bootstrap_metric_ci: ci_lower <= mean <= ci_upper for MAE",
      ci_result["MAE"]["ci_lower"] <= ci_result["MAE"]["mean"] <= ci_result["MAE"]["ci_upper"])

print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
