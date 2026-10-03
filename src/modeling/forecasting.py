from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "employees_clean.csv"
MODELS_DIR = ROOT / "models" / "forecasting"
REPORTS_DIR = ROOT / "reports" / "forecasting"

ForecastTarget = Literal["headcount", "hires", "exits", "attrition_rate"]


def load_workforce_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed workforce dataset and validate required columns."""
    df = pd.read_csv(path)

    required = {"employee_id", "attrition_flag"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def build_monthly_workforce_series(
    df: pd.DataFrame,
    start_date: str = "2021-01-01",
    periods: int = 48,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a reproducible synthetic monthly workforce time series.

    The IBM HR dataset is a point-in-time snapshot without hire or exit
    dates. This function creates a deterministic monthly timeline for
    portfolio forecasting and preserves the observed attrition total.
    """
    if periods < 24:
        raise ValueError("periods must be at least 24.")

    rng = np.random.default_rng(random_state)
    df = df.copy()

    employee_count = len(df)
    total_exits = int(df["attrition_flag"].sum())
    dates = pd.date_range(start=start_date, periods=periods, freq="MS")

    exit_weights = rng.dirichlet(np.ones(periods) * 2)
    exits = np.floor(exit_weights * total_exits).astype(int)

    remaining_exits = total_exits - exits.sum()
    if remaining_exits > 0:
        chosen_months = rng.choice(
            np.arange(periods),
            size=remaining_exits,
            replace=True,
        )
        for month in chosen_months:
            exits[month] += 1

    base_hires = max(3, int(round(total_exits / periods)))
    trend = np.linspace(-1.0, 2.0, periods)
    seasonality = 1.5 * np.sin(np.arange(periods) * 2 * np.pi / 12)

    hires = np.maximum(
        0,
        np.round(
            rng.poisson(base_hires, size=periods) + trend + seasonality
        ).astype(int),
    )

    headcount = np.zeros(periods, dtype=int)
    headcount[0] = max(
        300,
        employee_count - int(np.sum(hires)) + int(np.sum(exits)),
    )

    for index in range(1, periods):
        headcount[index] = max(
            1,
            headcount[index - 1] + hires[index] - exits[index],
        )

    workforce = pd.DataFrame(
        {
            "month": dates,
            "headcount": headcount,
            "hires": hires,
            "exits": exits,
        }
    )

    workforce["attrition_rate"] = (
        workforce["exits"]
        / workforce["headcount"].shift(1).replace(0, np.nan)
    ).fillna(0.0)

    workforce["net_change"] = workforce["hires"] - workforce["exits"]

    return workforce


def calculate_forecast_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE for forecast validation."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted values must have equal length.")
    if len(actual_values) == 0:
        raise ValueError("Cannot calculate metrics for empty arrays.")

    errors = actual_values - predicted_values

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    nonzero_actual = actual_values != 0
    if nonzero_actual.any():
        mape = float(
            np.mean(
                np.abs(
                    errors[nonzero_actual]
                    / actual_values[nonzero_actual]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_test_split_time_series(
    series: pd.Series,
    test_periods: int = 6,
) -> tuple[pd.Series, pd.Series]:
    """Split ordered time-series data into train and validation periods."""
    if test_periods <= 0:
        raise ValueError("test_periods must be positive.")
    if len(series) <= test_periods:
        raise ValueError("Series is too short for this test split.")

    return (
        series.iloc[:-test_periods].copy(),
        series.iloc[-test_periods:].copy(),
    )


def forecast_baseline(
    train: pd.Series,
    horizon: int,
) -> np.ndarray:
    """Forecast by repeating the most recent observed value."""
    return np.repeat(float(train.iloc[-1]), horizon)


def forecast_holt_winters(
    train: pd.Series,
    horizon: int,
) -> np.ndarray:
    """Forecast using Holt-Winters exponential smoothing."""
    train = train.astype(float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ValueWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        use_seasonality = len(train) >= 24

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if use_seasonality else None,
            seasonal_periods=12 if use_seasonality else None,
            initialization_method="estimated",
        )

        fitted_model = model.fit(optimized=True)
        forecast = fitted_model.forecast(horizon)

    return np.asarray(forecast, dtype=float)


def select_best_model(
    train: pd.Series,
    validation: pd.Series,
) -> tuple[str, pd.DataFrame]:
    """Compare baseline and Holt-Winters models on validation data."""
    horizon = len(validation)

    predictions = {
        "baseline": forecast_baseline(train, horizon),
        "holt_winters": forecast_holt_winters(train, horizon),
    }

    rows = []
    for model_name, forecast in predictions.items():
        rows.append(
            {
                "model": model_name,
                **calculate_forecast_metrics(validation, forecast),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(
        by=["mae", "rmse"],
        ascending=True,
    )

    return str(metrics.iloc[0]["model"]), metrics


def make_future_forecast(
    series: pd.Series,
    model_name: str,
    horizon: int,
) -> np.ndarray:
    """Refit the selected model on all observed history and forecast forward."""
    if model_name == "baseline":
        return forecast_baseline(series, horizon)

    if model_name == "holt_winters":
        return forecast_holt_winters(series, horizon)

    raise ValueError(f"Unknown model: {model_name}")


def run_forecasting_pipeline(
    data_path: Path = DATA_PATH,
    test_periods: int = 6,
    forecast_horizon: int = 12,
) -> dict[str, object]:
    """Build workforce history, validate models, forecast, and save artifacts."""
    raw_data = load_workforce_data(data_path)
    workforce = build_monthly_workforce_series(raw_data)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    workforce_path = REPORTS_DIR / "monthly_workforce_metrics.csv"
    workforce.to_csv(workforce_path, index=False)

    forecast_frames = []
    metric_frames = []

    targets = ["headcount", "hires", "exits", "attrition_rate"]

    for target in targets:
        series = workforce.set_index("month")[target].astype(float)
        train, validation = train_test_split_time_series(
            series,
            test_periods=test_periods,
        )

        best_model, validation_metrics = select_best_model(
            train,
            validation,
        )

        validation_metrics.insert(0, "target", target)
        validation_metrics.insert(1, "validation_periods", test_periods)
        metric_frames.append(validation_metrics)

        future_values = make_future_forecast(
            series=series,
            model_name=best_model,
            horizon=forecast_horizon,
        )

        future_months = pd.date_range(
            start=series.index[-1] + pd.offsets.MonthBegin(1),
            periods=forecast_horizon,
            freq="MS",
        )

        forecast_frames.append(
            pd.DataFrame(
                {
                    "month": future_months,
                    "target": target,
                    "model": best_model,
                    "forecast": future_values,
                }
            )
        )

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)

    count_targets = ["headcount", "hires", "exits"]
    forecasts.loc[
        forecasts["target"].isin(count_targets),
        "forecast",
    ] = forecasts.loc[
        forecasts["target"].isin(count_targets),
        "forecast",
    ].clip(lower=0)

    forecasts.loc[
        forecasts["target"] == "attrition_rate",
        "forecast",
    ] = forecasts.loc[
        forecasts["target"] == "attrition_rate",
        "forecast",
    ].clip(lower=0, upper=1)

    forecast_path = REPORTS_DIR / "forecast_results.csv"
    metrics_path = REPORTS_DIR / "forecast_metrics.csv"

    forecasts.to_csv(forecast_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    metadata = {
        "module": "2.5 Workforce Forecasting",
        "source_data": str(data_path),
        "history_periods": int(len(workforce)),
        "validation_periods": int(test_periods),
        "forecast_horizon": int(forecast_horizon),
        "models_compared": ["baseline", "holt_winters"],
        "targets": targets,
        "artifacts": {
            "monthly_workforce_metrics": str(workforce_path),
            "forecast_results": str(forecast_path),
            "forecast_metrics": str(metrics_path),
        },
        "data_limitation": (
            "The source IBM HR dataset is a point-in-time snapshot without "
            "employment event dates. The monthly history is synthetic, "
            "deterministic, and intended solely for demonstration."
        ),
    }

    metadata_path = MODELS_DIR / "forecast_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {
        "workforce": workforce,
        "forecasts": forecasts,
        "metrics": metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    results = run_forecasting_pipeline()

    print("\nMonthly workforce metrics:")
    print(results["workforce"].tail())

    print("\nValidation metrics:")
    print(results["metrics"].to_string(index=False))

    print("\nForecast artifacts saved to:")
    for name, path in results["metadata"]["artifacts"].items():
        print(f"- {name}: {path}")