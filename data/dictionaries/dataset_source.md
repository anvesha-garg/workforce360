# Dataset Source

## Base Dataset

**Name:** IBM HR Analytics Employee Attrition & Performance

**Source:** Kaggle  
**Dataset page:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Local File

Place the downloaded CSV file at:

```text
data/raw/ibm_hr_attrition.csv
```

The original file may be named:

```text
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Rename it to `ibm_hr_attrition.csv` after downloading.

## Dataset Usage Notice

This dataset is used only for portfolio, learning, analytics, and model-development purposes.

The dataset is treated as a sample HR dataset. It must not be interpreted as real employee data for production decisions.

## Version Notes

- Expected records: approximately 1,470
- Expected columns: 35
- Target column: `Attrition`
- Target values: `Yes` and `No`

## Git Policy

Raw data is intentionally excluded from Git through `.gitignore`.

Only dataset documentation, source code, schemas, notebooks, tests, and non-sensitive sample files are committed.