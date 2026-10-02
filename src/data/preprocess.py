from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "employees_clean.csv"
)

REQUIRED_COLUMNS = {
    "EmployeeNumber",
    "Age",
    "Attrition",
    "Department",
    "JobRole",
    "Gender",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
}


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Verify that required IBM HR columns exist before preprocessing.

    Args:
        df: Raw IBM HR dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


def create_tenure_bucket(years: pd.Series) -> pd.Series:
    """
    Group employee tenure into readable categories.

    Args:
        years: Series containing years at company.

    Returns:
        Categorical series containing tenure buckets.
    """
    return pd.cut(
        years,
        bins=[-1, 1, 3, 5, 10, float("inf")],
        labels=[
            "0-1 years",
            "2-3 years",
            "4-5 years",
            "6-10 years",
            "11+ years",
        ],
    )


def clean_hr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean IBM HR data and create Workforce360 analytical features.

    The original IBM columns are retained where possible. This function
    adds standardized snake_case fields and derived features for
    dashboards, KPIs, and later ML models.
    """
    validate_required_columns(df)

    clean_df = df.copy()

    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    numeric_columns = clean_df.select_dtypes(include="number").columns
    categorical_columns = clean_df.select_dtypes(
    include=["object", "string"]).columns
    for column in numeric_columns:
        clean_df[column] = clean_df[column].fillna(clean_df[column].median())

    for column in categorical_columns:
        clean_df[column] = clean_df[column].fillna("Unknown")

    clean_df["employee_id"] = (
        "EMP" + clean_df["EmployeeNumber"].astype(int).astype(str).str.zfill(4)
    )

    clean_df["attrition_flag"] = clean_df["Attrition"].map({"Yes": 1, "No": 0})
    clean_df["overtime_flag"] = clean_df["OverTime"].map({"Yes": 1, "No": 0})

    clean_df["age_group"] = pd.cut(
        clean_df["Age"],
        bins=[0, 25, 35, 45, 55, float("inf")],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
        include_lowest=True,
    )

    clean_df["tenure_bucket"] = create_tenure_bucket(clean_df["YearsAtCompany"])

    clean_df["years_since_promotion"] = clean_df["YearsSinceLastPromotion"]

    clean_df["income_vs_department_avg"] = (
        clean_df["MonthlyIncome"]
        / clean_df.groupby("Department")["MonthlyIncome"].transform("mean")
    ).round(3)

    clean_df["is_below_department_income_avg"] = (
        clean_df["income_vs_department_avg"] < 1
    ).astype(int)
    
    clean_df = clean_df.rename(
        columns={
            "Age": "age",
            "Attrition": "attrition",
            "BusinessTravel": "business_travel",
            "DailyRate": "daily_rate",
            "Department": "department",
            "DistanceFromHome": "distance_from_home",
            "Education": "education",
            "EducationField": "education_field",
            "EmployeeCount": "employee_count",
            "EmployeeNumber": "source_employee_number",
            "EnvironmentSatisfaction": "environment_satisfaction",
            "Gender": "gender",
            "HourlyRate": "hourly_rate",
            "JobInvolvement": "job_involvement",
            "JobLevel": "job_level",
            "JobRole": "job_role",
            "JobSatisfaction": "job_satisfaction",
            "MaritalStatus": "marital_status",
            "MonthlyIncome": "monthly_income",
            "MonthlyRate": "monthly_rate",
            "NumCompaniesWorked": "num_companies_worked",
            "Over18": "over_18",
            "OverTime": "overtime",
            "PercentSalaryHike": "percent_salary_hike",
            "PerformanceRating": "performance_rating",
            "RelationshipSatisfaction": "relationship_satisfaction",
            "StandardHours": "standard_hours",
            "StockOptionLevel": "stock_option_level",
            "TotalWorkingYears": "total_working_years",
            "TrainingTimesLastYear": "training_times_last_year",
            "WorkLifeBalance": "work_life_balance",
            "YearsAtCompany": "years_at_company",
            "YearsInCurrentRole": "years_in_current_role",
            "YearsSinceLastPromotion": "years_since_last_promotion_source",
            "YearsWithCurrManager": "years_with_curr_manager",
        }
    )

    return clean_df


def save_processed_data(
    df: pd.DataFrame, output_path: str | Path | None = None
) -> Path:
    """
    Save cleaned Workforce360 data to the processed-data directory.

    Args:
        df: Cleaned Workforce360 DataFrame.
        output_path: Optional custom output path.

    Returns:
        Path where the processed CSV was saved.
    """
    destination = Path(output_path) if output_path else DEFAULT_PROCESSED_DATA_PATH

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)

    return destination