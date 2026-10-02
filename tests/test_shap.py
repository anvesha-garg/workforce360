import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.analytics.insights import (
    compute_shap_values,
    get_global_feature_importance,
    get_top_factors_for_employee,
    get_transformed_feature_names,
    transform_features,
)
from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data
from src.modeling.attrition import (
    load_attrition_model,
    prepare_attrition_data,
    split_attrition_data,
)


@pytest.fixture(scope="module")
def fitted_model_and_data():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    X, y = prepare_attrition_data(clean_df)
    X_train, X_test, _, _ = split_attrition_data(X, y)

    model = load_attrition_model()

    return model, X_train, X_test


def test_loaded_model_is_pipeline(fitted_model_and_data):
    model, _, _ = fitted_model_and_data

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_transformed_feature_names_match_transformed_data(fitted_model_and_data):
    model, _, X_test = fitted_model_and_data

    transformed_df = transform_features(model, X_test.head(5))
    feature_names = get_transformed_feature_names(model)

    assert list(transformed_df.columns) == feature_names
    assert transformed_df.shape[1] == len(feature_names)
    assert transformed_df.shape[0] == 5


def test_compute_shap_values_has_expected_shape(fitted_model_and_data):
    model, X_train, X_test = fitted_model_and_data

    shap_values, transformed_df = compute_shap_values(
        model=model,
        X_background=X_train.head(100),
        X_explain=X_test.head(5),
    )

    values = np.asarray(shap_values.values)

    assert transformed_df.shape[0] == 5
    assert values.shape[0] == 5
    assert values.shape[1] == transformed_df.shape[1]


def test_get_top_factors_for_employee(fitted_model_and_data):
    model, X_train, X_test = fitted_model_and_data

    shap_values, transformed_df = compute_shap_values(
        model=model,
        X_background=X_train.head(100),
        X_explain=X_test.head(5),
    )

    factors = get_top_factors_for_employee(
        shap_values=shap_values,
        transformed_features=transformed_df,
        employee_position=0,
        top_n=5,
    )

    expected_columns = {
        "feature",
        "feature_value",
        "shap_value",
        "impact_direction",
        "absolute_impact",
    }

    assert len(factors) == 5
    assert expected_columns.issubset(factors.columns)
    assert factors["absolute_impact"].is_monotonic_decreasing


def test_get_global_feature_importance(fitted_model_and_data):
    model, X_train, X_test = fitted_model_and_data

    shap_values, _ = compute_shap_values(
        model=model,
        X_background=X_train.head(100),
        X_explain=X_test.head(10),
    )

    importance = get_global_feature_importance(
        shap_values=shap_values,
        top_n=10,
    )

    assert len(importance) == 10
    assert set(importance.columns) == {
        "feature",
        "mean_absolute_shap_value",
    }
    assert importance["mean_absolute_shap_value"].is_monotonic_decreasing