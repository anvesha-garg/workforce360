# Workforce360 Architecture

This document describes the high-level architecture of the Workforce360 platform.

## Data Flow Overview

1. **Data Sources**
   - CSV / Excel files (e.g., IBM HR Analytics dataset).
   - Synthetic HR data (for demo and extended fields).
   - Optional: PostgreSQL or other relational database in production.

2. **Data Ingestion & Processing**
   - Python scripts in `src/data/` load raw data from `data/raw/`.
   - Preprocessing and feature engineering are performed using Pandas/NumPy.
   - Cleaned data is saved to `data/processed/` and/or loaded into a SQL database.

3. **Analytics & Machine Learning**
   - **Analytics Layer** (`src/analytics/`):
     - Computes HR KPIs (attrition rate, average salary, tenure, etc.).
   - **Modeling Layer** (`src/modeling/`):
     - Attrition prediction models (Logistic Regression, Random Forest, XGBoost).
     - Employee segmentation (K-Means clustering).
     - Salary prediction (regression models).
     - Workforce forecasting (Prophet / time-series models).
   - Model explainability via SHAP for attrition and other models.

4. **BI & Visualization**
   - **Power BI**:
     - Connects to processed data / SQL.
     - Provides executive dashboards with interactive filters.
   - **Streamlit App** (`app/`):
     - Multipage application for:
       - Executive Overview
       - Workforce Analytics
       - Employee Segmentation
       - Attrition Prediction
       - Compensation Intelligence
       - Workforce Forecast
       - AI HR Copilot

5. **AI HR Copilot**
   - Built with LangChain + LLM.
   - Capabilities:
     - Answer natural-language questions about HR metrics.
     - Generate SQL queries from natural language (text-to-SQL).
     - Summarize workforce insights and generate executive reports.

6. **Deployment**
   - Streamlit Community Cloud:
     - Connected to this GitHub repository.
     - Automatic redeploy on push to `main`.
   - Secrets (LLM API keys, DB credentials) managed via `.streamlit/secrets.toml`.

## Component Diagram (Logical)

```text
                  DATA SOURCES
        ┌───────────┼────────────┐
        ↓           ↓            ↓
      CSV        Excel       PostgreSQL
        └───────────┼────────────┘
                    ↓
              DATA PIPELINE
                    ↓
             Python + SQL
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
      Analytics           ML Layer
      Pandas/SQL       Scikit-learn/XGBoost/Prophet
          ↓                   ↓
          └─────────┬─────────┘
                    ↓
              Power BI
                    ↓
          ┌─────────┴──────────┐
          ↓                    ↓
    BI Dashboards          Streamlit App
                               ↓
                         AI HR Copilot
                               ↓
                         LLM + Text-to-SQL
                               ↓
                         Insights / Reports
```

A visual diagram will be added later in this document.

## Modules

- `src/data/` – Data loading, validation, preprocessing.
- `src/analytics/` – KPI computation and insight generation.
- `src/modeling/` – ML models for attrition, segmentation, salary, forecasting.
- `src/ai_copilot/` – LLM integration, text-to-SQL, report generation.
- `app/` – Streamlit multipage UI.
- `powerbi/` – Power BI report definitions.
- `sql/` – Database schema and reusable queries.

## Future Enhancements

- Role-based access control.
- Automated data refresh pipelines.
- Model monitoring and retraining workflows.
- Integration with real HRIS systems.