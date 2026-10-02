from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data, save_processed_data


def main() -> None:
    """Load raw data, create Workforce360 features, and save processed data."""
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)
    output_path = save_processed_data(clean_df)

    print(f"Processed rows: {len(clean_df)}")
    print(f"Processed columns: {len(clean_df.columns)}")
    print(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    main()