import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data
from src.modeling.salary import (
    build_salary_pipeline,
    evaluate_salary_model,
    load_salary_model,
    prepare_salary_data,
    predict_salary_range,
    save_salary_model,
    split_salary_data,
)


@pytest.fixture(scope="module")
def salary_model_and_data():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    X, y = prepare_salary_data(clean_df)
    X_train, X_test, y_train, y_test = split_salary_data(X, y)

    model = build_salary_pipeline()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def test_prepare_salary_data(salary_model_and_data):
    _, X_train, X_test, y_train, y_test = salary_model_and_data

    assert not X_train.empty
    assert not X_test.empty
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.isna().sum().sum() == 0


def test_salary_pipeline_can_fit_and_predict(salary_model_and_data):
    model, _, X_test, _, _ = salary_model_and_data

    assert isinstance(model, Pipeline)

    predictions = model.predict(X_test.head(5))

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_evaluate_salary_model(salary_model_and_data):
    model, _, X_test, _, y_test = salary_model_and_data

    metrics = evaluate_salary_model(model, X_test, y_test)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1


def test_predict_salary_range(salary_model_and_data):
    model, _, X_test, _, _ = salary_model_and_data

    result = predict_salary_range(model, X_test.head(1))

    assert set(result) == {
        "point_estimate",
        "lower_bound",
        "upper_bound",
    }

    assert result["lower_bound"] < result["point_estimate"]
    assert result["point_estimate"] < result["upper_bound"]


def test_save_and_load_salary_model(tmp_path, salary_model_and_data):
    model, _, X_test, _, y_test = salary_model_and_data

    metrics = evaluate_salary_model(model, X_test, y_test)
    model_path = tmp_path / "salary_model.joblib"

    saved_path = save_salary_model(
        model=model,
        metrics=metrics,
        path=model_path,
    )

    assert saved_path.exists()

    artifact = load_salary_model(saved_path)

    assert artifact["target"] == "monthly_income"
    assert set(artifact["metrics"]) == {"mae", "rmse", "r2"}