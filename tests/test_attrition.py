import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data
from src.modeling.attrition import (
    FEATURE_COLUMNS,
    build_models,
    evaluate_classifier,
    load_attrition_model,
    prepare_attrition_data,
    select_best_model,
    split_attrition_data,
    train_attrition_models,
)


@pytest.fixture(scope="module")
def attrition_data():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)
    return prepare_attrition_data(clean_df)


def test_prepare_attrition_data_returns_expected_features(attrition_data):
    X, y = attrition_data

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_split_attrition_data_is_stratified(attrition_data):
    X, y = attrition_data

    X_train, X_test, y_train, y_test = split_attrition_data(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

    overall_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert abs(overall_rate - train_rate) < 0.02
    assert abs(overall_rate - test_rate) < 0.02


def test_build_models_returns_three_pipelines(attrition_data):
    _, y = attrition_data

    models = build_models(y)

    assert set(models.keys()) == {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }

    assert all(isinstance(model, Pipeline) for model in models.values())


def test_logistic_regression_pipeline_can_fit_and_predict(attrition_data):
    X, y = attrition_data

    X_train, X_test, y_train, _ = split_attrition_data(X, y)
    model = build_models(y_train)["Logistic Regression"]

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_evaluate_classifier_returns_expected_metrics(attrition_data):
    X, y = attrition_data

    X_train, X_test, y_train, y_test = split_attrition_data(X, y)
    model = build_models(y_train)["Logistic Regression"]

    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "threshold",
    }

    assert expected_keys.issubset(metrics.keys())
    assert 0 <= metrics["roc_auc"] <= 1
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_train_models_and_select_best_model(attrition_data):
    X, y = attrition_data

    X_train, X_test, y_train, y_test = split_attrition_data(X, y)

    fitted_models, comparison_df, detailed_metrics = train_attrition_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    best_model_name, best_model = select_best_model(
        fitted_models,
        comparison_df,
    )

    assert len(fitted_models) == 3
    assert len(comparison_df) == 3
    assert set(detailed_metrics.keys()) == set(fitted_models.keys())
    assert best_model_name in fitted_models
    assert isinstance(best_model, Pipeline)


def test_saved_attrition_model_loads_successfully():
    model = load_attrition_model()

    assert isinstance(model, Pipeline)