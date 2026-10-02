import pandas as pd

from src.data.load import EXPECTED_COLUMNS, load_raw_ibm_hr, validate_ibm_hr_data


def test_validate_ibm_hr_data_accepts_valid_minimal_data():
    data = {}

    for column in EXPECTED_COLUMNS:
        if column in {"Attrition", "OverTime"}:
            data[column] = ["No"]
        elif column in {
            "BusinessTravel",
            "Department",
            "EducationField",
            "Gender",
            "JobRole",
            "MaritalStatus",
            "Over18",
        }:
            data[column] = ["Test"]
        else:
            data[column] = [1]

    df = pd.DataFrame(data)

    validate_ibm_hr_data(df)


def test_load_raw_ibm_hr_has_expected_columns():
    df = load_raw_ibm_hr()

    assert not df.empty
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert set(df["Attrition"].unique()).issubset({"Yes", "No"})