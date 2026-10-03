import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data
from src.modeling.segmentation import (
    assign_cluster,
    fit_kmeans,
    prepare_segmentation_features,
    save_segmentation_model,
)


@pytest.fixture(scope="module")
def segmentation_data():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    X, features = prepare_segmentation_features(clean_df)

    return X, features


def test_prepare_segmentation_features(segmentation_data):
    X, features = segmentation_data

    assert not X.empty
    assert len(features) >= 3
    assert list(X.columns) == features
    assert X.isna().sum().sum() == 0


def test_fit_kmeans_returns_valid_model(segmentation_data):
    X, _ = segmentation_data

    model, evaluation_df, silhouette_scores = fit_kmeans(
        X=X.head(300),
        k_range=(3, 5),
    )

    assert isinstance(model, KMeans)
    assert len(evaluation_df) == 3
    assert len(silhouette_scores) == 3
    assert model.n_clusters in evaluation_df["k"].tolist()
    assert evaluation_df["silhouette_score"].between(-1, 1).all()


def test_assign_cluster(segmentation_data):
    X, features = segmentation_data

    model, _, _ = fit_kmeans(
        X=X.head(300),
        k_range=(3, 4),
    )

    scaler = StandardScaler()
    scaler.fit(X.head(300))
    model.scaler = scaler

    labels = assign_cluster(
        model=model,
        df_new=X.head(10),
        features=features,
    )

    assert len(labels) == 10
    assert set(labels).issubset(set(range(model.n_clusters)))


def test_save_segmentation_model(tmp_path, segmentation_data):
    X, features = segmentation_data

    model, _, _ = fit_kmeans(
        X=X.head(300),
        k_range=(3, 4),
    )

    scaler = StandardScaler()
    scaler.fit(X.head(300))
    model.scaler = scaler

    model_path = tmp_path / "segmentation_model.joblib"

    saved_path = save_segmentation_model(
        model=model,
        scaler=scaler,
        features=features,
        path=model_path,
    )

    assert saved_path.exists()