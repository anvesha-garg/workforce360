import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.analytics.kpis import workforce_kpi_summary
from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data

st.set_page_config(
    page_title="Workforce360 HR Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Workforce360 HR Analytics Dashboard")
st.caption(
    "Interactive workforce KPIs, attrition insights, and employee-level risk explanations."
)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    raw_df = load_raw_ibm_hr()
    return clean_hr_data(raw_df)


df = load_clean_data()

st.sidebar.header("Filters")

departments = ["All departments"] + sorted(df["department"].unique().tolist())
selected_department = st.sidebar.selectbox(
    "Department",
    departments,
)

job_roles = ["All job roles"] + sorted(df["job_role"].unique().tolist())
selected_job_role = st.sidebar.selectbox(
    "Job role",
    job_roles,
)

filtered_df = df.copy()

if selected_department != "All departments":
    filtered_df = filtered_df[
        filtered_df["department"] == selected_department
    ]

if selected_job_role != "All job roles":
    filtered_df = filtered_df[
        filtered_df["job_role"] == selected_job_role
    ]

st.subheader("Workforce KPIs")

kpis = workforce_kpi_summary(filtered_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Total Employees",
    value=f"{kpis['total_employees']:,}",
)

col2.metric(
    label="Active Employees",
    value=f"{kpis['active_employees']:,}",
)

col3.metric(
    label="Attrition Rate",
    value=f"{kpis['attrition_rate']:.2f}%",
)

col4.metric(
    label="Average Monthly Salary",
    value=f"${kpis['average_monthly_salary']:,.0f}",
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    label="Average Tenure",
    value=f"{kpis['average_tenure_years']:.1f} years",
)

col6.metric(
    label="Average Job Satisfaction",
    value=f"{kpis['average_job_satisfaction']:.2f} / 4",
)

col7.metric(
    label="Recent Promotion Rate",
    value=f"{kpis['recent_promotion_rate']:.2f}%",
)

col8.metric(
    label="Overtime Rate",
    value=f"{kpis['overtime_percentage']:.2f}%",
)

st.divider()

st.subheader("Attrition Overview")

attrition_counts = (
    filtered_df["attrition_flag"]
    .value_counts()
    .rename_axis("Attrition Status")
    .reset_index(name="Employees")
)

attrition_counts["Attrition Status"] = attrition_counts[
    "Attrition Status"
].map(
    {
        0: "Stayed",
        1: "Left",
    }
)

st.bar_chart(
    attrition_counts.set_index("Attrition Status"),
    use_container_width=True,
)

st.divider()

st.subheader("Attrition Risk Prediction")

st.info(
    "Employee-level attrition-risk prediction and SHAP explanations will be added "
    "in the next dashboard iteration."
)