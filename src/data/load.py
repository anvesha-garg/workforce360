from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ibm_hr_attrition.csv"

EXPECTED_COLUMNS = {
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
}

NUMERIC_COLUMNS = {
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
}

CATEGORICAL_COLUMNS = {
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "Over18",
    "OverTime",
}


def validate_ibm_hr_data(df: pd.DataFrame) -> None:
    """
    Validate that the loaded IBM HR dataset has expected columns,
    acceptable values, and non-empty records.

    Raises:
        ValueError: If any validation check fails.
    """
    if df.empty:
        raise ValueError("Dataset is empty. Check the CSV file and its path.")

    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    invalid_attrition = set(df["Attrition"].dropna().unique()) - {"Yes", "No"}
    if invalid_attrition:
        raise ValueError(
            "Attrition contains unexpected values: "
            f"{sorted(invalid_attrition)}. Expected only 'Yes' or 'No'."
        )

    invalid_overtime = set(df["OverTime"].dropna().unique()) - {"Yes", "No"}
    if invalid_overtime:
        raise ValueError(
            "OverTime contains unexpected values: "
            f"{sorted(invalid_overtime)}. Expected only 'Yes' or 'No'."
        )

    non_numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            "The following columns should be numeric but are not: "
            f"{', '.join(sorted(non_numeric_columns))}"
        )


def load_raw_ibm_hr(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and validate the IBM HR Attrition CSV dataset.

    Args:
        file_path: Optional custom path to the CSV file.
                   If omitted, the default raw-data path is used.

    Returns:
        A validated pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file cannot be found.
        ValueError: If validation fails.
    """
    csv_path = Path(file_path) if file_path else DEFAULT_RAW_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Download the IBM HR dataset and save it as:\n"
            "data/raw/ibm_hr_attrition.csv"
        )

    df = pd.read_csv(csv_path)
    validate_ibm_hr_data(df)

    return df




