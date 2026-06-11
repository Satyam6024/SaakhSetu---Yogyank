"""
Yogyank Entitlement Score safer baseline.

This script is intentionally review-oriented rather than production-complete.
It trains a deterministic, leakage-conscious model pipeline and saves the
preprocessing, model, schema, validation metrics, and version metadata needed
for reproducible scoring review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_VERSION = "yogyank_baseline_v1.0.0"
RANDOM_STATE = 42

DEFAULT_DATA_CANDIDATES = (
    "farmer_scoring_sample_yogyank_round1.csv",
    "farmer_scoring_sample_yogyank_round1_final.csv",
)

ID_COLUMN = "farmer_id"
TIME_COLUMN = "application_year"
TARGET_COLUMN = "target_entitlement_score"
FUTURE_OUTCOME_COLUMNS = ["defaulted_in_next_12_months"]

NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score",
]

CATEGORICAL_FEATURES = [
    "district",
    "crop_type",
    "pm_kisan_status",
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TRAINING_REQUIRED_COLUMNS = [
    ID_COLUMN,
    TIME_COLUMN,
    TARGET_COLUMN,
    *FEATURE_COLUMNS,
]

MONITORING_SLICES = [
    "crop_type",
    "district",
    "pm_kisan_status",
    "irrigation_type",
    "landholding_band",
]

FEATURE_LABELS = {
    "land_area_acres": "Land area",
    "historical_repayment_score": "Historical repayment score",
    "annual_income_inr": "Annual income",
    "liability_ratio_pct": "Liability ratio",
    "rainfall_deviation_pct": "Rainfall deviation",
    "ndvi_score": "NDVI vegetation score",
    "district": "District",
    "crop_type": "Crop type",
    "pm_kisan_status": "PM-Kisan status",
    "irrigation_type": "Irrigation type",
    "land_ownership": "Land ownership",
    "soil_type": "Soil type",
    "sales_channel": "Sales channel",
}


def resolve_data_path(path: str | None) -> Path:
    if path:
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return data_path

    for candidate in DEFAULT_DATA_CANDIDATES:
        data_path = Path(candidate)
        if data_path.exists():
            return data_path

    expected = ", ".join(DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError(f"No input CSV found. Tried: {expected}")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def load_training_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in TRAINING_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values")

    return df


def out_of_time_split(
    df: pd.DataFrame, cutoff_year: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    years = sorted(df[TIME_COLUMN].dropna().unique().tolist())
    if len(years) < 2:
        raise ValueError("Out-of-time validation requires at least two application years")

    cutoff = int(cutoff_year if cutoff_year is not None else max(years))
    train_df = df[df[TIME_COLUMN] < cutoff].copy()
    valid_df = df[df[TIME_COLUMN] >= cutoff].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError(
            "Invalid cutoff year. Need non-empty train rows before cutoff and "
            "non-empty validation rows at or after cutoff."
        )

    return train_df, valid_df, cutoff


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "prediction_mean": float(np.mean(y_pred)),
        "actual_mean": float(np.mean(y_true)),
    }


def add_landholding_band(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["landholding_band"] = pd.cut(
        work["land_area_acres"],
        bins=[-np.inf, 1, 2, 5, 10, np.inf],
        labels=[
            "marginal_<=1",
            "small_1_to_2",
            "semi_medium_2_to_5",
            "medium_5_to_10",
            "large_>10",
        ],
    ).astype(str)
    return work


def compute_slice_metrics(
    valid_df: pd.DataFrame, predictions: np.ndarray
) -> dict[str, Any]:
    work = add_landholding_band(valid_df)
    work["_prediction"] = predictions
    work["_error"] = work["_prediction"] - work[TARGET_COLUMN]
    work["_absolute_error"] = work["_error"].abs()

    rows: list[dict[str, Any]] = []
    for slice_col in MONITORING_SLICES:
        if slice_col not in work.columns:
            continue

        for value, group in work.groupby(slice_col, dropna=False, observed=False):
            rows.append(
                {
                    "slice": slice_col,
                    "value": str(value),
                    "n": int(len(group)),
                    "actual_mean": float(group[TARGET_COLUMN].mean()),
                    "prediction_mean": float(group["_prediction"].mean()),
                    "mean_error": float(group["_error"].mean()),
                    "mae": float(group["_absolute_error"].mean()),
                }
            )

    return {
        "method": "Validation residual summary by monitoring slice.",
        "warning": (
            "These are diagnostic slices, not a fairness certification. "
            "Production monitoring should add outcome, approval, and drift metrics."
        ),
        "slices": rows,
    }


def target_leakage_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for col in FUTURE_OUTCOME_COLUMNS:
        if col not in df.columns:
            diagnostics[col] = {
                "present": False,
                "used_as_feature": False,
            }
            continue

        corr = None
        if df[col].nunique(dropna=True) > 1:
            corr = float(df[col].corr(df[TARGET_COLUMN]))
        diagnostics[col] = {
            "present": True,
            "used_as_feature": False,
            "correlation_with_target": corr,
            "risk": "future outcome unavailable at scoring time",
        }
    return diagnostics


def build_reason_baselines(train_df: pd.DataFrame) -> dict[str, Any]:
    numeric_baselines = {
        col: float(train_df[col].median()) for col in NUMERIC_FEATURES
    }
    categorical_baselines: dict[str, str | None] = {}
    for col in CATEGORICAL_FEATURES:
        mode = train_df[col].mode(dropna=True)
        categorical_baselines[col] = None if mode.empty else str(mode.iloc[0])

    return {
        "numeric_baselines": numeric_baselines,
        "categorical_baselines": categorical_baselines,
    }


def coerce_scoring_frame(record: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(record, pd.DataFrame):
        frame = record.copy()
    elif isinstance(record, pd.Series):
        frame = pd.DataFrame([record.to_dict()])
    else:
        frame = pd.DataFrame([record])

    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Scoring record is missing required model features: {missing}")

    return frame[FEATURE_COLUMNS].copy()


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (np.floating,)):
        return f"{float(value):.2f}"
    return str(value)


def make_reason_text(
    feature: str,
    observed_value: Any,
    baseline_value: Any,
    delta: float,
) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    direction = "increased" if delta >= 0 else "reduced"
    return (
        f"{label}={format_value(observed_value)} versus baseline "
        f"{format_value(baseline_value)} {direction} the score by about "
        f"{abs(delta):.1f} points."
    )


def top_reason_codes(
    pipeline: Pipeline,
    record: dict[str, Any] | pd.Series | pd.DataFrame,
    baselines: dict[str, Any],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    frame = coerce_scoring_frame(record)
    if len(frame) != 1:
        raise ValueError("Reason codes are produced one farmer record at a time")

    full_prediction = float(pipeline.predict(frame)[0])
    row = frame.iloc[0]
    contributions: list[dict[str, Any]] = []

    for feature in FEATURE_COLUMNS:
        baseline_value = (
            baselines["numeric_baselines"].get(feature)
            if feature in NUMERIC_FEATURES
            else baselines["categorical_baselines"].get(feature)
        )
        if baseline_value is None:
            continue

        perturbed = frame.copy()
        perturbed.loc[perturbed.index[0], feature] = baseline_value
        perturbed_prediction = float(pipeline.predict(perturbed)[0])
        delta = full_prediction - perturbed_prediction
        if np.isclose(delta, 0.0):
            continue

        contributions.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "observed_value": row[feature],
                "baseline_value": baseline_value,
                "estimated_impact_points": float(delta),
                "direction": "positive" if delta >= 0 else "negative",
                "reason": make_reason_text(feature, row[feature], baseline_value, delta),
            }
        )

    contributions.sort(key=lambda item: abs(item["estimated_impact_points"]), reverse=True)
    return [
        {"rank": rank, **item}
        for rank, item in enumerate(contributions[:top_n], start=1)
    ]


def score_farmer(
    record: dict[str, Any] | pd.Series | pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    top_n: int = 3,
) -> dict[str, Any]:
    artifacts_path = Path(artifacts_dir)
    pipeline = joblib.load(artifacts_path / "model_pipeline.joblib")
    reference = load_json(artifacts_path / "reason_code_reference.json")
    version_info = load_json(artifacts_path / "model_version.json")

    frame = coerce_scoring_frame(record)
    prediction = float(pipeline.predict(frame)[0])

    farmer_id = None
    if isinstance(record, pd.DataFrame) and ID_COLUMN in record.columns and len(record) == 1:
        farmer_id = record.iloc[0][ID_COLUMN]
    elif isinstance(record, pd.Series) and ID_COLUMN in record.index:
        farmer_id = record[ID_COLUMN]
    elif isinstance(record, dict):
        farmer_id = record.get(ID_COLUMN)

    return {
        "farmer_id": farmer_id,
        "model_version": version_info["model_version"],
        "score": round(prediction, 2),
        "reason_codes": top_reason_codes(
            pipeline,
            record,
            {
                "numeric_baselines": reference["numeric_baselines"],
                "categorical_baselines": reference["categorical_baselines"],
            },
            top_n=top_n,
        ),
    }


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    try:
        return pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        return []


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return json_ready(value.tolist())
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_and_save(
    data_path: Path,
    artifacts_dir: Path,
    cutoff_year: int | None = None,
) -> dict[str, Any]:
    df = load_training_data(data_path)
    train_df, valid_df, cutoff = out_of_time_split(df, cutoff_year=cutoff_year)

    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    predictions = pipeline.predict(valid_df[FEATURE_COLUMNS])
    metrics = compute_metrics(valid_df[TARGET_COLUMN], predictions)
    leakage_checks = target_leakage_diagnostics(df)
    slice_metrics = compute_slice_metrics(valid_df, predictions)
    reason_baselines = build_reason_baselines(train_df)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifacts_dir / "model_pipeline.joblib")

    feature_schema = {
        "model_version": MODEL_VERSION,
        "scoring_required_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "time_split_column": TIME_COLUMN,
        "excluded_from_model": {
            "identifier": [ID_COLUMN],
            "target": [TARGET_COLUMN],
            "time_split_only": [TIME_COLUMN],
            "future_outcome_leakage": FUTURE_OUTCOME_COLUMNS,
        },
        "transformed_feature_names": transformed_feature_names(pipeline),
        "unknown_category_behavior": "OneHotEncoder(handle_unknown='ignore')",
    }

    version_info = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "script_sha256": file_sha256(Path(__file__)),
        "training_data_file": str(data_path),
        "training_data_sha256": file_sha256(data_path),
        "git_commit": git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "random_state": RANDOM_STATE,
    }

    validation_metrics = {
        "model_version": MODEL_VERSION,
        "split_strategy": "out_of_time",
        "cutoff_year": cutoff,
        "train_years": sorted(train_df[TIME_COLUMN].unique().tolist()),
        "validation_years": sorted(valid_df[TIME_COLUMN].unique().tolist()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "metrics": metrics,
        "target_leakage_checks": leakage_checks,
    }

    training_metadata = {
        "model_version": MODEL_VERSION,
        "objective": "Predict a bank-agnostic Yogyank Entitlement Score.",
        "not_policy": (
            "This model does not output probability of default, eligibility, "
            "bank cutoffs, grade mappings, or approval decisions."
        ),
        "model_family": "HistGradientBoostingRegressor inside sklearn Pipeline",
        "features_used": FEATURE_COLUMNS,
        "features_excluded": {
            "farmer_id": "identifier only",
            "application_year": "used only for out-of-time validation split",
            "defaulted_in_next_12_months": "future outcome unavailable at scoring time",
            TARGET_COLUMN: "training target",
        },
        "preprocessing": {
            "numeric": "median imputation",
            "categorical": "most-frequent imputation plus one-hot encoding",
            "fit_scope": "preprocessing fitted on training rows only via Pipeline",
        },
        "validation": validation_metrics,
    }

    reason_reference = {
        "model_version": MODEL_VERSION,
        "method": (
            "For a single farmer, predict once as submitted, then replace each "
            "feature one at a time with the training baseline and rank the "
            "largest prediction changes."
        ),
        "warning": (
            "Reason codes are local model diagnostics, not causal explanations "
            "and not policy decisions."
        ),
        "feature_labels": FEATURE_LABELS,
        **reason_baselines,
    }

    example_reason_codes = score_farmer_in_memory(
        pipeline=pipeline,
        record=valid_df.iloc[0],
        baselines=reason_baselines,
        version=MODEL_VERSION,
    )

    save_json(artifacts_dir / "feature_schema.json", feature_schema)
    save_json(artifacts_dir / "model_version.json", version_info)
    save_json(artifacts_dir / "validation_metrics.json", validation_metrics)
    save_json(artifacts_dir / "slice_metrics.json", slice_metrics)
    save_json(artifacts_dir / "training_metadata.json", training_metadata)
    save_json(artifacts_dir / "reason_code_reference.json", reason_reference)
    save_json(artifacts_dir / "example_reason_codes.json", example_reason_codes)

    return validation_metrics


def score_farmer_in_memory(
    pipeline: Pipeline,
    record: dict[str, Any] | pd.Series | pd.DataFrame,
    baselines: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    frame = coerce_scoring_frame(record)
    prediction = float(pipeline.predict(frame)[0])

    farmer_id = None
    if isinstance(record, pd.Series) and ID_COLUMN in record.index:
        farmer_id = record[ID_COLUMN]
    elif isinstance(record, dict):
        farmer_id = record.get(ID_COLUMN)
    elif isinstance(record, pd.DataFrame) and ID_COLUMN in record.columns and len(record) == 1:
        farmer_id = record.iloc[0][ID_COLUMN]

    return {
        "farmer_id": farmer_id,
        "model_version": version,
        "score": round(prediction, 2),
        "reason_codes": top_reason_codes(pipeline, record, baselines, top_n=3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or use the safer Yogyank baseline model."
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to training CSV. Defaults to the provided assessment CSV name.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for saved model, schema, metrics, and metadata.",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=None,
        help="First validation year. Defaults to the latest application_year.",
    )
    parser.add_argument(
        "--score-json",
        default=None,
        help="Optional path to a JSON farmer record to score using saved artifacts.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    if args.score_json:
        with Path(args.score_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = score_farmer(payload, artifacts_dir=artifacts_dir, top_n=3)
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
        return

    data_path = resolve_data_path(args.data)
    validation_metrics = train_and_save(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        cutoff_year=args.cutoff_year,
    )

    print("Training complete.")
    print(f"Input data: {data_path}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Validation split: train < {validation_metrics['cutoff_year']}, valid >= {validation_metrics['cutoff_year']}")
    print(json.dumps(validation_metrics["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
