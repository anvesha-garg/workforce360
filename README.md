# Workforce360 — AI-Powered Workforce Intelligence & HR Analytics Platform

An end-to-end HR analytics platform integrating Power BI, SQL, machine learning, predictive analytics, and Generative AI to analyze workforce trends, predict attrition, segment employees, forecast workforce requirements, detect anomalies, and provide conversational business insights.

## Problem Statement

HR teams struggle to:

- Understand workforce trends and risks in a unified view.
- Predict which employees are likely to leave and why.
- Identify employee segments with distinct needs and risks.
- Forecast future headcount, hiring, and attrition.
- Interact with HR data using natural language instead of complex dashboards.

Workforce360 addresses these gaps by combining BI, data science, and GenAI into a single decision-support platform.

## High-Level Features

- **Executive Workforce Dashboard**: KPIs, trends, and filters for HR leaders.
- **Attrition Prediction**: ML models (Logistic Regression, Random Forest, XGBoost) with SHAP-based explainability.
- **Employee Segmentation**: Unsupervised clustering to discover employee groups.
- **Compensation Intelligence**: Salary prediction and equity analysis.
- **Workforce Forecasting**: Time-series forecasts for headcount, hiring, and attrition.
- **AI HR Copilot**: Conversational interface for HR analytics.
- **Natural Language → SQL**: Ask questions in English and get answers from the underlying data.
- **AI-Generated Reports**: Automated executive summaries and insights.
- **Anomaly Detection**: Identify unusual patterns in attrition, overtime, hiring, etc.
- **Employee Risk Dashboard**: Department-level and employee-level risk views.

## Tech Stack

- **Language**: Python
- **Data & Analytics**: Pandas, NumPy, SQL (PostgreSQL/SQLite)
- **Machine Learning**: Scikit-learn, XGBoost, SHAP, Prophet
- **BI**: Power BI
- **App Framework**: Streamlit (multipage app)
- **GenAI**: LangChain + LLM (for AI HR Copilot & text-to-SQL)
- **Deployment**: Streamlit Community Cloud + GitHub
- **CI/CD**: GitHub Actions (linting, tests)

## Repo Structure

```text
workforce360/
├─ data/
│  ├─ raw/              # Raw datasets (e.g., IBM HR, synthetic extensions)
│  ├─ processed/        # Cleaned and feature-engineered data
│  └─ dictionaries/     # Data dictionary, schema docs
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_modeling_attrition.ipynb
│  └─ ...
├─ src/
│  ├─ data/             # Data loading and preprocessing
│  ├─ modeling/         # ML models (attrition, segmentation, salary, forecasting)
│  ├─ analytics/        # KPIs and insight generation
│  └─ ai_copilot/       # LLM client, text-to-SQL, report generator
├─ sql/
│  ├─ schema.sql
│  └─ queries/
├─ powerbi/
│  └─ Workforce360.pbix
├─ app/
│  ├─ app.py            # Streamlit entrypoint
│  └─ pages/            # Multipage app (one file per page)
├─ tests/
│  ├─ test_data.py
│  ├─ test_modeling.py
│  └─ test_app.py
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ docs/
│  ├─ architecture.md
│  └─ deployment.md
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## How to Run Locally

```bash
# Clone the repository
git clone [https://github.com/yourusername/workforce360.git](https://github.com/yourusername/workforce360.git)
cd workforce360

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

Access the app at `http://localhost:8501`.

## Deployment

The app is deployed on **Streamlit Community Cloud**, connected to this GitHub repository. Every push to `main` triggers an automatic redeployment.

- Live demo: *(link to be added)*
- Deployment details: see [`docs/deployment.md`](docs/deployment.md).

## License

MIT