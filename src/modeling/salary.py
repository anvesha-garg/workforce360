from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
SALARY_MODEL_PATH = MODELS_DIR / "salary_model.joblib"

TARGET_COLUMN = "monthly_income"

CATEGORICAL_FEATURES = [
    "department",
    "job_role",
    "education_field",
    "gender",
    "overtime_flag",
]

NUMERIC_FEATURES = [
    "age",
    "distance_from_home",
    "total_working_years",
    "years_at_company",
    "years_in_current_role",
    "years_since_promotion",
    "job_level",
    "performance_rating",
    "job_satisfaction",
    "work_life_balance",
]

SALARY_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def prepare_salary_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for monthly-income prediction.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = set(SALARY_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    X = df[SALARY_FEATURES].copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(
        X[NUMERIC_FEATURES].median(numeric_only=True)
    )

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("Unknown")

    return X, y


def split_salary_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split salary data into training and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def build_salary_pipeline() -> Pipeline:
    """Build a Gradient Boosting regression pipeline for salary prediction."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_salary_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared for a fitted salary model."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def predict_salary_range(
    model: Pipeline,
    employee_row: pd.DataFrame,
    range_multiplier: float = 0.10,
) -> dict[str, float]:
    """
    Return a salary point estimate and an uncertainty-based range.

    The range is a simple percentage band around the point prediction. It is
    useful for dashboard communication, but it is not a formal statistical
    prediction interval.
    """
    if employee_row.empty:
        raise ValueError("employee_row must contain at least one employee.")

    point_estimate = float(model.predict(employee_row)[0])
    range_width = point_estimate * range_multiplier

    return {
        "point_estimate": point_estimate,
        "lower_bound": point_estimate - range_width,
        "upper_bound": point_estimate + range_width,
    }


def save_salary_model(
    model: Pipeline,
    metrics: dict[str, float],
    path: Path = SALARY_MODEL_PATH,
) -> Path:
    """Save the fitted salary model and evaluation metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "metrics": metrics,
        "features": SALARY_FEATURES,
        "target": TARGET_COLUMN,
    }

    joblib.dump(artifact, path)
    return path


def load_salary_model(
    path: Path = SALARY_MODEL_PATH,
) -> dict:
    """Load a saved salary-model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Salary model not found: {path}")

    return joblib.load(path)