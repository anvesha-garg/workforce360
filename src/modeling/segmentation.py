from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DEFAULT_SEGMENTATION_FEATURES = [
    "age",
    "monthly_income",
    "years_at_company",
    "total_working_years",
    "years_since_promotion",
    "job_satisfaction",
    "overtime_flag",
]

MODELS_DIR = Path("models")
SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_model.joblib"


def prepare_segmentation_features(
    df: pd.DataFrame,
    features: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select and clean numeric features for employee segmentation.

    Returns:
        - Feature DataFrame with missing values filled using column medians
        - List of features actually used
    """
    requested_features = list(features or DEFAULT_SEGMENTATION_FEATURES)
    available_features = [
        feature for feature in requested_features if feature in df.columns
    ]

    if len(available_features) < 3:
        raise ValueError(
            "At least three segmentation features must be available in the data."
        )

    feature_df = df[available_features].copy()

    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(
            feature_df[column],
            errors="coerce",
        )

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    return feature_df, available_features


def fit_kmeans(
    X: pd.DataFrame,
    k_range: tuple[int, int] = (3, 6),
    random_state: int = 42,
) -> tuple[KMeans, pd.DataFrame, list[float]]:
    """
    Fit KMeans models across a range of cluster counts.

    Cluster count is selected using the highest silhouette score.

    Returns:
        - Best fitted KMeans model
        - Evaluation DataFrame with k, inertia, and silhouette score
        - Silhouette scores in the same order as k_range
    """
    if X.empty:
        raise ValueError("Input feature matrix is empty.")

    X_scaled = StandardScaler().fit_transform(X)

    min_k, max_k = k_range

    if min_k < 2 or max_k < min_k:
        raise ValueError("k_range must be a valid range starting at 2 or higher.")

    max_k = min(max_k, len(X) - 1)

    results: list[dict[str, float | int]] = []
    silhouette_scores: list[float] = []
    fitted_models: dict[int, KMeans] = {}

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)

        fitted_models[k] = kmeans
        silhouette_scores.append(float(silhouette))

        results.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette_score": float(silhouette),
            }
        )

    evaluation_df = pd.DataFrame(results)
    best_k = int(
        evaluation_df.loc[
            evaluation_df["silhouette_score"].idxmax(),
            "k",
        ]
    )

    return fitted_models[best_k], evaluation_df, silhouette_scores


def assign_cluster(
    model: KMeans,
    df_new: pd.DataFrame,
    features: Iterable[str] | None = None,
) -> np.ndarray:
    """
    Assign cluster labels to new employee records using a fitted KMeans model.

    The function scales the new data with the scaler stored inside the model
    artifact and returns the predicted cluster number for each row.
    """
    feature_list = list(
        features
        or getattr(model, "feature_names_in_", DEFAULT_SEGMENTATION_FEATURES)
    )

    missing_features = set(feature_list) - set(df_new.columns)

    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{', '.join(sorted(missing_features))}"
        )

    X_new = df_new[feature_list].copy()

    for column in X_new.columns:
        X_new[column] = pd.to_numeric(X_new[column], errors="coerce")

    X_new = X_new.fillna(X_new.median(numeric_only=True))

    scaler: StandardScaler = model.scaler
    X_scaled = scaler.transform(X_new)

    return model.predict(X_scaled)


def save_segmentation_model(
    model: KMeans,
    scaler: StandardScaler,
    features: list[str],
    path: Path = SEGMENTATION_MODEL_PATH,
) -> Path:
    """Save the KMeans model, scaler, and feature list as one artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": features,
    }

    joblib.dump(artifact, path)
    return path


def load_segmentation_model(
    path: Path = SEGMENTATION_MODEL_PATH,
) -> dict:
    """Load a saved segmentation artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Segmentation model not found: {path}")

    return joblib.load(path)