from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "attrition_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "attrition_model_metadata.json"

TARGET_COLUMN = "attrition_flag"

FEATURE_COLUMNS = [
    "age",
    "business_travel",
    "department",
    "distance_from_home",
    "education",
    "education_field",
    "environment_satisfaction",
    "gender",
    "job_involvement",
    "job_level",
    "job_role",
    "job_satisfaction",
    "marital_status",
    "monthly_income",
    "num_companies_worked",
    "overtime",
    "percent_salary_hike",
    "performance_rating",
    "relationship_satisfaction",
    "stock_option_level",
    "total_working_years",
    "training_times_last_year",
    "work_life_balance",
    "years_at_company",
    "years_in_current_role",
    "years_since_promotion",
    "years_with_curr_manager",
]

NUMERIC_FEATURES = [
    "age",
    "distance_from_home",
    "education",
    "environment_satisfaction",
    "job_involvement",
    "job_level",
    "job_satisfaction",
    "monthly_income",
    "num_companies_worked",
    "percent_salary_hike",
    "performance_rating",
    "relationship_satisfaction",
    "stock_option_level",
    "total_working_years",
    "training_times_last_year",
    "work_life_balance",
    "years_at_company",
    "years_in_current_role",
    "years_since_promotion",
    "years_with_curr_manager",
]

CATEGORICAL_FEATURES = [
    "business_travel",
    "department",
    "education_field",
    "gender",
    "job_role",
    "marital_status",
    "overtime",
]


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that the DataFrame contains all expected modeling columns."""
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Missing required modeling columns: "
            f"{', '.join(missing_columns)}"
        )


def prepare_attrition_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and binary attrition target y.

    Excludes direct identifiers and avoids derived target-leakage fields such as
    employee_id, attrition text label, income comparison, and risk outputs.
    """
    _require_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("attrition_flag must contain only 0 and 1.")

    return X, y


def split_attrition_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split for attrition modeling."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a mixed-type preprocessing transformer.

    Numeric fields are median-imputed; they are additionally standardized for
    Logistic Regression. Categorical fields are mode-imputed and one-hot encoded.
    """
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Build the three candidate classification pipelines."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgboost = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }


def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate a binary classifier using threshold-sensitive and ROC metrics."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)),
            4,
        ),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "threshold": threshold,
    }


def train_attrition_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Train candidate models and return fitted models, comparison table, and
    detailed metrics. Selection is based on ROC-AUC, then F1 score.
    """
    candidate_models = build_models(y_train)
    fitted_models: dict[str, Pipeline] = {}
    detailed_metrics: dict[str, dict[str, Any]] = {}

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        detailed_metrics[model_name] = evaluate_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
        )

    comparison_rows = []

    for model_name, metrics in detailed_metrics.items():
        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(
            by=["roc_auc", "f1_score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return fitted_models, comparison_df, detailed_metrics


def select_best_model(
    fitted_models: dict[str, Pipeline],
    comparison_df: pd.DataFrame,
) -> tuple[str, Pipeline]:
    """Select the top-ranked model from the comparison table."""
    best_model_name = str(comparison_df.iloc[0]["model"])
    return best_model_name, fitted_models[best_model_name]


def save_attrition_model(
    model: Pipeline,
    model_name: str,
    metrics: dict[str, Any],
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save a fitted attrition pipeline and lightweight metadata."""
    resolved_model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    resolved_metadata_path = (
        Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    )

    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, resolved_model_path)

    metadata = {
        "model_name": model_name,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset_note": (
            "Trained on IBM HR Analytics portfolio data. "
            "Not for real employment decisions."
        ),
    }

    with resolved_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return resolved_model_path, resolved_metadata_path


def load_attrition_model(
    model_path: str | Path | None = None,
) -> Pipeline:
    """Load a serialized attrition model pipeline."""
    resolved_model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Attrition model not found at: {resolved_model_path}. "
            "Train and save a model first."
        )

    return joblib.load(resolved_model_path)