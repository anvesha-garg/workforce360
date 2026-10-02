import pandas as pd

from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data, create_tenure_bucket


def test_clean_hr_data_has_expected_columns():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    expected_columns = {
        "employee_id",
        "source_employee_number",
        "attrition_flag",
        "overtime_flag",
        "age_group",
        "tenure_bucket",
        "years_since_promotion",
        "monthly_income",
        "years_at_company",
        "job_satisfaction",
        "work_life_balance",
    }

    assert expected_columns.issubset(clean_df.columns)


def test_clean_hr_data_has_no_missing_values_in_key_columns():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    key_columns = [
        "employee_id",
        "age",
        "department",
        "job_role",
        "monthly_income",
        "attrition_flag",
        "overtime_flag",
        "years_at_company",
        "job_satisfaction",
    ]

    assert clean_df[key_columns].isna().sum().sum() == 0


def test_binary_flags_have_expected_values():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    assert set(clean_df["attrition_flag"].unique()).issubset({0, 1})
    assert set(clean_df["overtime_flag"].unique()).issubset({0, 1})


def test_employee_id_is_unique():
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    assert clean_df["employee_id"].is_unique


def test_create_tenure_bucket():
    years = pd.Series([0, 1, 2, 4, 6, 12])
    buckets = create_tenure_bucket(years)

    assert list(buckets.astype(str)) == [
        "0-1 years",
        "0-1 years",
        "2-3 years",
        "4-5 years",
        "6-10 years",
        "11+ years",
    ]