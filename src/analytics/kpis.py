import pandas as pd


def _require_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error if a KPI's required columns are missing."""
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


def total_employees(df: pd.DataFrame) -> int:
    """Return the total employee record count."""
    return int(len(df))


def active_employees(df: pd.DataFrame) -> int:
    """
    Return employees currently considered active.

    For this historical IBM dataset, `attrition_flag = 0` is used as an
    analytical proxy for an employee who did not leave the organization.
    """
    _require_columns(df, {"attrition_flag"})

    return int((df["attrition_flag"] == 0).sum())


def attrition_rate(df: pd.DataFrame) -> float:
    """Return observed attrition rate as a percentage from 0 to 100."""
    _require_columns(df, {"attrition_flag"})

    if df.empty:
        return 0.0

    return round(float(df["attrition_flag"].mean() * 100), 2)


def average_salary(df: pd.DataFrame) -> float:
    """Return average monthly income."""
    _require_columns(df, {"monthly_income"})

    if df.empty:
        return 0.0

    return round(float(df["monthly_income"].mean()), 2)


def average_tenure(df: pd.DataFrame) -> float:
    """Return average years at the company."""
    _require_columns(df, {"years_at_company"})

    if df.empty:
        return 0.0

    return round(float(df["years_at_company"].mean()), 2)


def average_job_satisfaction(df: pd.DataFrame) -> float:
    """Return average job-satisfaction rating on the 1-4 source-data scale."""
    _require_columns(df, {"job_satisfaction"})

    if df.empty:
        return 0.0

    return round(float(df["job_satisfaction"].mean()), 2)


def promotion_rate(df: pd.DataFrame, recent_years: int = 2) -> float:
    """
    Return proxy promotion rate: percentage promoted within `recent_years`.

    The IBM dataset has no direct promotion-event flag, so this calculates
    employees with years_since_promotion less than or equal to recent_years.
    """
    _require_columns(df, {"years_since_promotion"})

    if df.empty:
        return 0.0

    rate = (df["years_since_promotion"] <= recent_years).mean() * 100
    return round(float(rate), 2)


def overtime_percentage(df: pd.DataFrame) -> float:
    """Return percentage of employees recorded as working overtime."""
    _require_columns(df, {"overtime_flag"})

    if df.empty:
        return 0.0

    return round(float(df["overtime_flag"].mean() * 100), 2)


def workforce_kpi_summary(df: pd.DataFrame) -> dict[str, int | float]:
    """Return all Workforce360 executive KPIs in one dictionary."""
    return {
        "total_employees": total_employees(df),
        "active_employees": active_employees(df),
        "attrition_rate": attrition_rate(df),
        "average_monthly_salary": average_salary(df),
        "average_tenure_years": average_tenure(df),
        "average_job_satisfaction": average_job_satisfaction(df),
        "recent_promotion_rate": promotion_rate(df),
        "overtime_percentage": overtime_percentage(df),
    }