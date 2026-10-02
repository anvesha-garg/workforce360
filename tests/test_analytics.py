import pandas as pd
import pytest

from src.analytics.kpis import (
    active_employees,
    attrition_rate,
    average_job_satisfaction,
    average_salary,
    average_tenure,
    overtime_percentage,
    promotion_rate,
    total_employees,
    workforce_kpi_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "attrition_flag": [0, 1, 0, 0],
            "monthly_income": [5000, 7000, 9000, 11000],
            "years_at_company": [1, 3, 5, 7],
            "job_satisfaction": [1, 2, 3, 4],
            "years_since_promotion": [0, 2, 3, 5],
            "overtime_flag": [0, 1, 1, 0],
        }
    )


def test_total_employees(sample_df):
    assert total_employees(sample_df) == 4


def test_active_employees(sample_df):
    assert active_employees(sample_df) == 3


def test_attrition_rate(sample_df):
    assert attrition_rate(sample_df) == 25.0


def test_average_salary(sample_df):
    assert average_salary(sample_df) == 8000.0


def test_average_tenure(sample_df):
    assert average_tenure(sample_df) == 4.0


def test_average_job_satisfaction(sample_df):
    assert average_job_satisfaction(sample_df) == 2.5


def test_promotion_rate(sample_df):
    assert promotion_rate(sample_df, recent_years=2) == 50.0


def test_overtime_percentage(sample_df):
    assert overtime_percentage(sample_df) == 50.0


def test_workforce_kpi_summary(sample_df):
    summary = workforce_kpi_summary(sample_df)

    assert summary["total_employees"] == 4
    assert summary["active_employees"] == 3
    assert summary["attrition_rate"] == 25.0
    assert summary["average_monthly_salary"] == 8000.0


def test_missing_required_column_raises_error():
    df = pd.DataFrame({"age": [25, 30]})

    with pytest.raises(ValueError, match="Missing required columns"):
        attrition_rate(df)