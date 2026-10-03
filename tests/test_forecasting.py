import numpy as np
import pandas as pd
import pytest

from src.modeling.forecasting import (
    build_monthly_workforce_series,
    calculate_forecast_metrics,
    forecast_baseline,
    load_workforce_data,
    train_test_split_time_series,
)


@pytest.fixture
def workforce_data():
    return pd.DataFrame(
        {
            "employee_id": ["EMP001", "EMP002", "EMP003", "EMP004"],
            "attrition_flag": [0, 1, 0, 1],
        }
    )


def test_load_workforce_data():
    df = load_workforce_data()

    assert not df.empty
    assert {"employee_id", "attrition_flag"}.issubset(df.columns)


def test_build_monthly_workforce_series(workforce_data):
    monthly = build_monthly_workforce_series(
        workforce_data,
        periods=24,
        random_state=42,
    )

    expected_columns = {
        "month",
        "headcount",
        "hires",
        "exits",
        "attrition_rate",
        "net_change",
    }

    assert len(monthly) == 24
    assert expected_columns.issubset(monthly.columns)
    assert pd.api.types.is_datetime64_any_dtype(monthly["month"])


def test_monthly_workforce_values_are_valid(workforce_data):
    monthly = build_monthly_workforce_series(
        workforce_data,
        periods=24,
        random_state=42,
    )

    assert (monthly["headcount"] > 0).all()
    assert (monthly["hires"] >= 0).all()
    assert (monthly["exits"] >= 0).all()
    assert monthly["attrition_rate"].between(0, 1).all()
    assert (
        monthly["net_change"]
        == monthly["hires"] - monthly["exits"]
    ).all()


def test_calculate_forecast_metrics():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])

    metrics = calculate_forecast_metrics(actual, predicted)

    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["mae"] == pytest.approx(7 / 3)
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0


def test_mape_handles_zero_actual_values():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([2.0, 8.0, 22.0])

    metrics = calculate_forecast_metrics(actual, predicted)

    assert np.isfinite(metrics["mape"])
    assert metrics["mape"] > 0


def test_time_series_split():
    series = pd.Series(
        range(12),
        index=pd.date_range("2024-01-01", periods=12, freq="MS"),
    )

    train, test = train_test_split_time_series(series, test_periods=3)

    assert len(train) == 9
    assert len(test) == 3
    assert train.index.max() < test.index.min()


def test_baseline_forecast_repeats_last_value():
    train = pd.Series([10.0, 12.0, 15.0])

    forecast = forecast_baseline(train, horizon=4)

    assert len(forecast) == 4
    assert np.allclose(forecast, [15.0, 15.0, 15.0, 15.0])