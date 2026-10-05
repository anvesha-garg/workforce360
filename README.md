# Workforce360

### Workforce intelligence for better people decisions

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20Workforce360-0F766E?style=for-the-badge)](https://workforce360-byuxtts2cxmf8qjedcqn4t.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-47-2EA44F?style=flat-square)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> Workforce360 is an end-to-end workforce intelligence and HR analytics platform that combines Business Intelligence, machine learning, explainable AI, compensation analysis, employee segmentation, and workforce forecasting in one interactive application.

## Live application

### [Launch Workforce360](https://workforce360-byuxtts2cxmf8qjedcqn4t.streamlit.app/)

Explore workforce KPIs, attrition risk, employee segments, compensation insights, salary predictions, workforce forecasts, what-if scenarios, and the free local AI HR Copilot.

---

## Why Workforce360?

HR teams often have access to large amounts of workforce data but lack a clear way to transform that data into useful decisions.

Workforce360 brings together:

- Descriptive analytics to understand current workforce health.
- Business Intelligence dashboards to monitor important workforce KPIs.
- Predictive analytics to identify potential attrition risk.
- Explainable machine learning to show what influences model predictions.
- Segmentation to group employees with similar workforce patterns.
- Compensation analysis to compare salary across workforce dimensions.
- Forecasting to support future headcount and retention planning.
- A free local AI Copilot to summarize workforce insights without paid API credits.

The goal is not to automate HR decisions. The goal is to help HR teams ask better questions, identify patterns earlier, and focus their attention where it can have the greatest impact.

---

## Key capabilities

### Workforce BI dashboard

The application includes an interactive Business Intelligence layer built with Streamlit and Plotly.

It provides:

- Total employee count.
- Active employee count.
- Attrition rate.
- Average monthly salary.
- Average tenure.
- Average job satisfaction.
- Recent promotion rate.
- Overtime percentage.
- Attrition comparisons by department.
- Interactive department and job-role filters.
- Downloadable analysis tables.

### Attrition prediction

The platform estimates employee attrition probability using a trained classification model.

Users can:

- Select an employee.
- View estimated attrition probability.
- See low, medium, or high-risk classification.
- Review supporting employee attributes.
- Prioritize supportive retention conversations.

### SHAP explainability

Workforce360 uses SHAP to make model outputs easier to understand.

The application shows:

- Factors that increase estimated attrition risk.
- Factors that reduce estimated attrition risk.
- Employee-level feature contributions.
- Global feature importance.
- Waterfall explanations for individual predictions.

SHAP values explain the model's behavior. They should not be interpreted as proven causes of employee attrition.

### Employee segmentation

Employees are grouped using KMeans clustering based on workforce characteristics such as:

- Career progression.
- Compensation.
- Job satisfaction.
- Workload.
- Tenure.
- Employment profile.

The segmentation dashboard includes:

- Segment sizes.
- Attrition rate by segment.
- Average segment profiles.
- Sample employees from each cluster.

### Compensation intelligence

The compensation module provides:

- Salary distribution analysis.
- Average and median salary.
- Total payroll.
- Department salary comparisons.
- Pay comparisons by gender, department, job role, or job level.
- Salary prediction for a hypothetical employee profile.

Salary predictions are decision-support references and should be reviewed alongside market benchmarks, internal equity, performance, and HR policy.

### Workforce forecasting

The forecasting module supports workforce planning through:

- Historical headcount trends.
- Hiring trends.
- Exit trends.
- Attrition-rate trends.
- Twelve-month forecast views.
- Model validation metrics.
- Projected headcount change.
- Downloadable forecast results.

Important: the source HR dataset is a point-in-time snapshot without operational hiring and termination dates. Therefore, the monthly history and forecast are synthetic planning demonstrations rather than production workforce forecasts.

### What-if simulator

The simulator allows users to change hypothetical employee attributes such as:

- Monthly income.
- Overtime status.
- Job satisfaction.
- Years at the company.
- Years since promotion.
- Age.

The model then estimates how the hypothetical profile's attrition probability changes.

This feature is designed for scenario analysis, not decisions about real individuals.

### Free local AI HR Copilot

Workforce360 includes a floating AI HR Copilot button in the bottom-right corner of the application.

The Copilot can answer questions about:

- Workforce size.
- Attrition.
- Headcount planning.
- Compensation.
- Engagement.
- Overtime.
- Promotion patterns.
- Workforce forecasts.
- Retention priorities.

The Copilot is implemented as a local, rule-based insight engine. It does not require:

- OpenAI credits.
- API keys.
- Paid cloud AI services.
- External LLM requests.
- Internet access while answering.

---

## Skills demonstrated

### Business Intelligence

- KPI design.
- Dashboard development.
- Interactive filtering.
- Data exploration.
- Drill-down analysis.
- Trend analysis.
- Workforce reporting.
- Decision-support visualization.
- Report-ready tables.
- CSV exports.
- Executive-oriented storytelling with data.

### Data analytics

- Data loading and validation.
- Data cleaning and preprocessing.
- Feature engineering.
- Exploratory data analysis.
- Grouped aggregations.
- Workforce metric calculation.
- Missing-value handling.
- Data quality checks.

### Machine learning

- Classification modeling.
- Attrition prediction.
- Regression modeling for salary estimation.
- KMeans clustering.
- Feature preprocessing pipelines.
- Train-test splitting.
- Stratified evaluation.
- Model persistence with Joblib.
- Model comparison and selection.

### Explainable AI

- SHAP-based feature attribution.
- Individual prediction explanations.
- Global feature importance.
- Waterfall plots.
- Responsible interpretation of model outputs.

### Time-series forecasting

- Monthly workforce series construction.
- Headcount forecasting.
- Hiring and exit forecasting.
- Attrition-rate forecasting.
- Baseline forecasting.
- Forecast validation.
- MAE, RMSE, and MAPE evaluation.
- Planning-oriented forecast communication.

### Software engineering

- Modular Python project structure.
- Reusable functions.
- Type hints.
- Caching with Streamlit.
- Error handling.
- Artifact loading.
- Cross-platform file paths.
- Git branching and merging.
- Automated testing.
- Deployment preparation.

### Product development

- User-centered dashboard design.
- Consistent visual language.
- Interactive workflows.
- Floating chat interface.
- Responsible-use messaging.
- Decision-support framing.
- Public cloud deployment.

---

## Technology stack

| Area | Tools |
|---|---|
| Language | Python |
| Data manipulation | pandas |
| Numerical computing | NumPy |
| Machine learning | scikit-learn |
| Explainable AI | SHAP |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Model persistence | Joblib |
| Testing | pytest |
| Version control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

---

## Project structure

```text
workforce360/
│
├── data/
│   ├── dictionaries/
│   │   ├── data_dictionary.md
│   │   └── dataset_source.md
│   ├── processed/
│   │   └── employees_clean.csv
│   └── raw/
│       └── ibm_hr_attrition.csv
│
├── models/
│   ├── attrition_model.joblib
│   ├── salary_model.joblib
│   └── segmentation_model.joblib
│
├── notebooks/
│   ├── 01_data_loading_and_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_attrition_model.ipynb
│   ├── 04_shap_explainability.ipynb
│   ├── 05_segmentation.ipynb
│   ├── 06_salary_model.ipynb
│   └── 07_workforce_forecasting.ipynb
│
├── reports/
│   ├── attrition_model_metrics.json
│   ├── salary_model_metrics.json
│   └── forecasting/
│       ├── monthly_workforce_metrics.csv
│       ├── forecast_results.csv
│       └── forecast_metrics.csv
│
├── src/
│   ├── ai_copilot/
│   │   └── llm_client.py
│   ├── analytics/
│   │   ├── insights.py
│   │   └── kpis.py
│   ├── app/
│   │   └── app.py
│   ├── data/
│   │   ├── load.py
│   │   └── preprocess.py
│   └── modeling/
│       ├── attrition.py
│       ├── forecasting.py
│       ├── salary.py
│       └── segmentation.py
│
├── tests/
│   ├── test_ai_copilot.py
│   ├── test_analytics.py
│   ├── test_attrition.py
│   ├── test_data.py
│   ├── test_forecasting.py
│   ├── test_load.py
│   ├── test_salary.py
│   ├── test_segmentation.py
│   └── test_shap.py
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Run locally

### 1. Clone the repository

```bash
git clone [https://github.com/anvesha-garg/workforce360.git](https://github.com/anvesha-garg/workforce360.git)
cd workforce360
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run src/app/app.py
```

The application should open in your browser automatically.

---

## Run the tests

Run the complete test suite:

```bash
python -m pytest tests -v
```

The project includes tests for:

- AI Copilot context generation.
- KPI calculations.
- Data cleaning.
- Dataset validation.
- Attrition modeling.
- Salary prediction.
- Employee segmentation.
- Forecasting utilities.
- SHAP explanations.
- Model persistence.

Expected result:

```text
47 passed
```

---

## Deployment

Workforce360 is deployed with Streamlit Community Cloud.

### Live application

[Open the deployed Workforce360 app](https://workforce360-byuxtts2cxmf8qjedcqn4t.streamlit.app/)

The deployed app runs from the `main` branch and uses:

```text
src/app/app.py
```

No paid AI API key is required for the Copilot.

---

## Responsible use

Workforce360 is a decision-support application, not an automated employment-decision system.

Do not use the application to:

- Automatically terminate or reject employees.
- Penalize or discipline employees.
- Make final employment decisions.
- Treat risk scores as facts about individuals.
- Treat SHAP values as proven causes.
- Use salary predictions without reviewing internal equity and market context.
- Use forecasts as guaranteed future outcomes.

Recommended use:

- Prioritize supportive conversations.
- Identify workforce patterns for further investigation.
- Explore possible planning scenarios.
- Combine model outputs with HR expertise and employee context.
- Regularly review model quality, data quality, and fairness.

---

## Data limitations

The project uses a public/sample IBM HR attrition dataset.

Important limitations include:

- The data is not a live HRIS feed.
- It represents a point-in-time workforce snapshot.
- It does not provide complete operational hiring and termination timelines.
- Forecast outputs are synthetic planning demonstrations.
- Historical patterns may not generalize to every organization.
- Model performance may change across departments, roles, and populations.
- Predictions should be monitored for fairness and unintended bias.

---

## Future improvements

Possible future enhancements include:

- Automated CI with GitHub Actions.
- Fairness and subgroup performance monitoring.
- Connection to a real HRIS or warehouse.
- Role-based access control.
- Live workforce data refresh.
- Confidence intervals for forecasts.
- More advanced natural-language question handling.
- Power BI executive reporting layer.
- Model monitoring and drift detection.
- Exportable executive PDF reports.

---

## Support and contribution

If you find a bug or have a suggestion:

1. Open a GitHub issue.
2. Include the error message or expected behavior.
3. Describe the steps needed to reproduce the issue.
4. Include your Python and package versions when relevant.

Pull requests are welcome for improvements to analytics, visualization, testing, documentation, and responsible-use safeguards.

---

## Author

Built as an end-to-end portfolio project demonstrating Business Intelligence, data analytics, machine learning, explainable AI, forecasting, software engineering, and product-focused dashboard development.