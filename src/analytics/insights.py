from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_preprocessor(model: Pipeline) -> Any:
    """
    Return the fitted preprocessing component from an attrition pipeline.

    Raises:
        ValueError: If the expected pipeline step is not present.
    """
    if "preprocessor" not in model.named_steps:
        raise ValueError("Model pipeline does not contain a 'preprocessor' step.")

    return model.named_steps["preprocessor"]


def get_classifier(model: Pipeline) -> Any:
    """
    Return the fitted classifier component from an attrition pipeline.

    Raises:
        ValueError: If the expected pipeline step is not present.
    """
    if "classifier" not in model.named_steps:
        raise ValueError("Model pipeline does not contain a 'classifier' step.")

    return model.named_steps["classifier"]


def get_transformed_feature_names(model: Pipeline) -> list[str]:
    """Return output feature names after fitted preprocessing and one-hot encoding."""
    preprocessor = get_preprocessor(model)
    return list(preprocessor.get_feature_names_out())


def transform_features(
    model: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform original input features into model-ready feature columns.

    Returns:
        A dense DataFrame with transformed feature names.
    """
    preprocessor = get_preprocessor(model)
    transformed = preprocessor.transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = get_transformed_feature_names(model)

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index,
    )


def get_linear_explainer(
    model: Pipeline,
    X_background: pd.DataFrame,
) -> shap.Explainer:
    """
    Create a SHAP linear explainer for the fitted Logistic Regression pipeline.

    The model is explained after its preprocessing stage, which ensures SHAP
    values refer to exactly the features supplied to the classifier.
    """
    classifier = get_classifier(model)

    if not hasattr(classifier, "coef_"):
        raise ValueError(
            "This function requires a linear classifier with a 'coef_' attribute."
        )

    transformed_background = transform_features(model, X_background)

    masker = shap.maskers.Independent(
        transformed_background,
        max_samples=min(100, len(transformed_background)),
    )

    return shap.LinearExplainer(
        classifier,
        masker,
    )


def compute_shap_values(
    model: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for rows in X_explain using a Logistic Regression pipeline.

    Returns:
        - SHAP Explanation object
        - DataFrame of preprocessed input features aligned to SHAP values
    """
    explainer = get_linear_explainer(model, X_background)
    transformed_explain = transform_features(model, X_explain)
    shap_values = explainer(transformed_explain)

    return shap_values, transformed_explain


def get_top_factors_for_employee(
    shap_values: shap.Explanation,
    transformed_features: pd.DataFrame,
    employee_position: int = 0,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the most influential SHAP contributions for a single employee.

    Positive SHAP values increase predicted attrition risk. Negative values
    decrease predicted attrition risk, relative to the model baseline.

    Args:
        shap_values: SHAP explanation object for one or more employees.
        transformed_features: Preprocessed model-input DataFrame.
        employee_position: Zero-based row position in the supplied explanation.
        top_n: Number of contributors to return.

    Returns:
        DataFrame sorted by absolute contribution magnitude.
    """
    if employee_position < 0 or employee_position >= len(transformed_features):
        raise IndexError("employee_position is outside the explanation range.")

    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        values = values[:, :, 1]

    employee_values = values[employee_position]

    factors = pd.DataFrame(
        {
            "feature": transformed_features.columns,
            "feature_value": transformed_features.iloc[employee_position].values,
            "shap_value": employee_values,
        }
    )

    factors["impact_direction"] = np.where(
        factors["shap_value"] >= 0,
        "increases_attrition_risk",
        "decreases_attrition_risk",
    )

    factors["absolute_impact"] = factors["shap_value"].abs()

    return (
        factors.sort_values("absolute_impact", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_global_feature_importance(
    shap_values: shap.Explanation,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return mean absolute SHAP value by transformed feature.

    Higher values mean the feature has greater average influence on model
    output across the explained employee sample.
    """
    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_values = np.abs(values).mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": shap_values.feature_names,
                "mean_absolute_shap_value": mean_abs_values,
            }
        )
        .sort_values("mean_absolute_shap_value", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )