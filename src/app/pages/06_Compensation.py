import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Compensation & Salary Prediction", page_icon="💰", layout="wide")

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "processed" / "employees_clean.csv"
MODEL_PATH = ROOT / "models" / "salary_model.joblib"
METRICS_PATH = ROOT / "reports" / "salary_model_metrics.json"

st.title("💰 Compensation & Salary Prediction")
st.caption("Analyze compensation distribution and estimate a fair market salary for an employee profile.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}

df = load_data()
model = load_model()
metrics = load_metrics()

salary_col = next(
    (c for c in ["monthly_income", "MonthlyIncome", "salary", "Salary", "annual_salary"]
     if c in df.columns),
    None,
)

if salary_col is None:
    st.error("No salary column found in the dataset.")
    st.stop()

df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")
df = df.dropna(subset=[salary_col])

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(df):,}")
c2.metric("Average Salary", f"₹{df[salary_col].mean():,.0f}")
c3.metric("Median Salary", f"₹{df[salary_col].median():,.0f}")
c4.metric("Total Payroll", f"₹{df[salary_col].sum():,.0f}")

st.divider()

# ---------------- Compensation analytics ----------------
st.subheader("Compensation Distribution")

left, right = st.columns([2, 1])

with left:
    fig = px.histogram(
        df,
        x=salary_col,
        nbins=40,
        title="Salary Distribution",
        labels={salary_col: "Salary"},
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

with right:
    group_col = next(
        (c for c in ["Department", "department", "JobRole", "job_role"] if c in df.columns),
        None,
    )
    if group_col:
        grouped = (
            df.groupby(group_col)[salary_col]
            .agg(["mean", "median", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        fig2 = px.bar(
            grouped,
            x="mean",
            y=group_col,
            orientation="h",
            title=f"Average Salary by {group_col}",
            labels={"mean": "Average Salary", group_col: group_col},
        )
        fig2.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No department or role column available for grouping.")

st.divider()

# ---------------- Salary prediction ----------------
st.subheader("Salary Prediction")
st.write("Enter an employee profile to estimate an appropriate salary using the trained salary model.")

model_features = getattr(model, "feature_names_in_", None)

numeric_defaults = {
    "Age": 35,
    "age": 35,
    "TotalWorkingYears": 10,
    "total_working_years": 10,
    "YearsAtCompany": 5,
    "years_at_company": 5,
    "YearsInCurrentRole": 3,
    "years_in_current_role": 3,
    "YearsSinceLastPromotion": 2,
    "years_since_last_promotion": 2,
    "YearsWithCurrManager": 4,
    "years_with_curr_manager": 4,
    "NumCompaniesWorked": 2,
    "num_companies_worked": 2,
    "DistanceFromHome": 10,
    "distance_from_home": 10,
    "Education": 3,
    "education": 3,
    "JobLevel": 2,
    "job_level": 2,
    "PerformanceRating": 3,
    "performance_rating": 3,
}

categorical_defaults = {
    "Department": "Research & Development",
    "department": "Research & Development",
    "JobRole": "Sales Executive",
    "job_role": "Sales Executive",
    "JobLevel": "2",
    "job_level": "2",
    "EducationField": "Life Sciences",
    "education_field": "Life Sciences",
    "Gender": "Male",
    "gender": "Male",
    "MaritalStatus": "Married",
    "marital_status": "Married",
    "BusinessTravel": "Travel_Rarely",
    "business_travel": "Travel_Rarely",
    "OverTime": "No",
    "over_time": "No",
}

input_features = list(model_features) if model_features is not None else []

if not input_features:
    st.warning(
        "The loaded model does not expose feature names. "
        "Prediction inputs cannot be generated safely."
    )
    st.stop()

with st.form("salary_prediction_form"):
    st.markdown("#### Employee Profile")

    form_cols = st.columns(3)
    user_input = {}

    for i, feature in enumerate(input_features):
        col = form_cols[i % 3]

        if feature in numeric_defaults or feature.lower() in {k.lower() for k in numeric_defaults}:
            default = numeric_defaults.get(feature, numeric_defaults.get(feature.lower(), 0))
            user_input[feature] = col.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                value=float(default),
                step=1.0,
            )
        elif feature in categorical_defaults or feature.lower() in {k.lower() for k in categorical_defaults}:
            default = categorical_defaults.get(feature, categorical_defaults.get(feature.lower(), ""))
            options = sorted(df[feature].dropna().astype(str).unique().tolist()) if feature in df.columns else [default]
            if default not in options:
                options.insert(0, default)
            user_input[feature] = col.selectbox(
                feature.replace("_", " ").title(),
                options,
                index=options.index(default),
            )
        else:
            user_input[feature] = col.text_input(feature.replace("_", " ").title(), "")

    submitted = st.form_submit_button("Predict Salary", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([user_input])

        for feature in input_features:
            if feature in input_df.columns and feature in df.columns:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce")

        prediction = model.predict(input_df)[0]
        prediction = max(float(prediction), 0)

        st.success(f"### Predicted Salary: ₹{prediction:,.0f}")

        if salary_col in df.columns:
            median_salary = df[salary_col].median()
            difference = prediction - median_salary
            percent = (difference / median_salary) * 100 if median_salary else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Dataset Median", f"₹{median_salary:,.0f}")
            m2.metric("Difference", f"₹{difference:,.0f}")
            m3.metric("Vs Median", f"{percent:+.1f}%")

        st.caption(
            "This estimate is a decision-support reference. "
            "Final compensation should also consider internal equity, performance, market benchmarks, and HR policy."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# ---------------- Model quality ----------------
st.subheader("Model Quality")

if metrics:
    metric_cols = st.columns(len(metrics))
    for col, (name, value) in zip(metric_cols, metrics.items()):
        if isinstance(value, (int, float)):
            col.metric(name.replace("_", " ").title(), f"{value:,.3f}")
        else:
            col.metric(name.replace("_", " ").title(), str(value))
else:
    st.info("Model metrics file not found. Run the salary-model notebook to generate it.")