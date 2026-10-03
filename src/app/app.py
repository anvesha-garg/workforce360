import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.analytics.insights import (
    compute_shap_values,
    get_top_factors_for_employee,
)
from src.analytics.kpis import workforce_kpi_summary
from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data
from src.modeling.attrition import (
    load_attrition_model,
    prepare_attrition_data,
    split_attrition_data,
)
from src.modeling.segmentation import (
    assign_cluster,
    load_segmentation_model,
    prepare_segmentation_features,
)

st.set_page_config(
    page_title="Workforce360",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] {
          font-family: 'Inter', sans-serif;
      }

      .block-container {
          padding-top: 1.5rem;
          padding-bottom: 3rem;
          max-width: 1300px;
      }

      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}

      h1 {
          font-size: 2.1rem !important;
          font-weight: 800 !important;
          letter-spacing: -0.03em;
          color: #0F172A !important;
      }

      h2 {
          font-size: 1.3rem !important;
          font-weight: 700 !important;
          color: #134E4A !important;
      }

      h3 {
          font-size: 1.05rem !important;
          font-weight: 650 !important;
          color: #334155 !important;
      }

      .w360-hero {
          background: linear-gradient(120deg, #0F766E 0%, #0EA5E9 60%, #6366F1 100%);
          border-radius: 22px;
          padding: 2rem 2.2rem;
          color: white;
          margin-bottom: 1.7rem;
          box-shadow: 0 18px 40px rgba(15, 118, 110, 0.20);
      }

      .w360-hero h1 {
          color: white !important;
          margin-bottom: 0.3rem;
      }

      .w360-hero p {
          font-size: 1rem;
          opacity: 0.94;
          margin: 0;
      }

      .w360-pill {
          display: inline-block;
          background: rgba(255,255,255,0.17);
          border: 1px solid rgba(255,255,255,0.28);
          padding: 0.28rem 0.75rem;
          border-radius: 999px;
          font-size: 0.76rem;
          font-weight: 600;
          margin-top: 1rem;
          margin-right: 0.4rem;
      }

      .w360-card {
          background: white;
          border: 1px solid #E2E8F0;
          border-radius: 18px;
          padding: 1.1rem 1.2rem;
          height: 100%;
          box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
      }

      .w360-feature {
          background: white;
          border-left: 5px solid #0F766E;
          border-radius: 14px;
          padding: 1rem 1.1rem;
          height: 100%;
          box-shadow: 0 6px 18px rgba(15,23,42,0.05);
      }

      .w360-feature h5 {
          margin: 0 0 0.3rem 0;
          font-size: 0.96rem;
          font-weight: 700;
          color: #134E4A;
      }

      .w360-feature p {
          font-size: 0.85rem;
          color: #475569;
          margin: 0;
          line-height: 1.45;
      }

      div[data-testid="stMetric"] {
          background: white;
          border: 1px solid #E2E8F0;
          border-radius: 16px;
          padding: 1rem;
          box-shadow: 0 6px 18px rgba(15,23,42,0.05);
      }

      div[data-testid="stMetricLabel"] p {
          color: #64748B !important;
          font-weight: 600 !important;
      }

      div[data-testid="stMetricValue"] {
          color: #0F172A !important;
          font-weight: 800 !important;
      }

      .stTabs [data-baseweb="tab-list"] {
          gap: 8px;
          background: #ECFDF5;
          padding: 6px;
          border-radius: 14px;
      }

      .stTabs [data-baseweb="tab"] {
          border-radius: 10px;
          padding: 0.55rem 1rem;
          font-weight: 600;
          color: #475569;
      }

      .stTabs [aria-selected="true"] {
          background: white !important;
          color: #0F766E !important;
          box-shadow: 0 3px 10px rgba(15,118,110,0.12);
      }

      .stButton > button {
          background: linear-gradient(90deg, #0F766E, #0EA5E9);
          color: white;
          border: none;
          border-radius: 12px;
          font-weight: 700;
          padding: 0.55rem 1.2rem;
      }

      .stButton > button:hover {
          filter: brightness(1.06);
      }

      .stDownloadButton > button {
          border-radius: 12px;
          font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    raw_df = load_raw_ibm_hr()
    return clean_hr_data(raw_df)


@st.cache_resource
def load_model():
    return load_attrition_model()


@st.cache_data
def load_evaluation_data():
    clean_df = clean_hr_data(load_raw_ibm_hr())
    X, y = prepare_attrition_data(clean_df)
    X_train, X_test, y_train, y_test = split_attrition_data(X, y)
    return X_train, X_test, y_train, y_test


@st.cache_resource
def load_salary_model():
    salary_model_path = ROOT / "models" / "salary_model.joblib"
    return joblib.load(salary_model_path)


@st.cache_data
def load_salary_metrics():
    metrics_path = ROOT / "reports" / "salary_model_metrics.json"

    if metrics_path.exists():
        with open(metrics_path, "r") as file:
            return json.load(file)

    return {}


def get_risk_tier(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


df = load_clean_data()
model = load_model()
X_train, X_test, y_train, y_test = load_evaluation_data()
salary_model = load_salary_model()
salary_metrics = load_salary_metrics()


@st.cache_resource
def load_segmentation_artifact():
    return load_segmentation_model()


segmentation_artifact = load_segmentation_artifact()
segmentation_model = segmentation_artifact["model"]
segmentation_scaler = segmentation_artifact["scaler"]
segmentation_features = segmentation_artifact["features"]


with st.sidebar:
    st.markdown("## 🧭 Workforce360")
    st.caption("Workforce Analytics, Forecasting & Decision Intelligence")

    st.divider()
    st.markdown("### Filters")

    departments = ["All departments"] + sorted(df["department"].unique().tolist())
    selected_department = st.selectbox("Department", departments)

    job_roles = ["All job roles"] + sorted(df["job_role"].unique().tolist())
    selected_job_role = st.selectbox("Job role", job_roles)

    st.divider()
    st.caption(
        "Workforce360 supports HR decision-making with explainable analytics. "
        "Predictions are decision-support tools, not automated employment decisions."
    )


filtered_df = df.copy()

if selected_department != "All departments":
    filtered_df = filtered_df[filtered_df["department"] == selected_department]

if selected_job_role != "All job roles":
    filtered_df = filtered_df[filtered_df["job_role"] == selected_job_role]


st.markdown(
    """
    <div class="w360-hero">
      <h1>Workforce360</h1>
      <p>Understand workforce health, attrition risk, employee segments, and compensation intelligence.</p>
      <div>
        <span class="w360-pill">Attrition Risk</span>
        <span class="w360-pill">Employee Segments</span>
        <span class="w360-pill">Compensation Intelligence</span>
        <span class="w360-pill">What-if Analysis</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### What you can do here")

feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

with feature_col1:
    st.markdown(
        """
        <div class="w360-feature">
          <h5>🎯 Predict Attrition</h5>
          <p>Identify employees with elevated resignation risk and understand the factors driving each prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col2:
    st.markdown(
        """
        <div class="w360-feature">
          <h5>🧭 Segment Employees</h5>
          <p>Group employees by career, compensation, engagement, and workload patterns for targeted HR action.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col3:
    st.markdown(
        """
        <div class="w360-feature">
          <h5>💰 Assess Compensation</h5>
          <p>Analyze salary distribution, compare pay across groups, and estimate market-aligned compensation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feature_col4:
    st.markdown(
        """
        <div class="w360-feature">
          <h5>🔍 Simulate Scenarios</h5>
          <p>Explore how changes in income, satisfaction, tenure, and workload affect estimated attrition risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

overview_tab, risk_tab, people_tab, segmentation_tab, compensation_tab, simulator_tab, model_tab = st.tabs(
    [
        "📊 Workforce overview",
        "🎯 Attrition risk",
        "👥 People to review",
        "🧭 Segmentation",
        "💰 Compensation",
        "🔍 What-if simulator",
        "📈 Model quality",
    ]
)


with overview_tab:
    if filtered_df.empty:
        st.warning("No employees match the selected filters.")
    else:
        kpis = workforce_kpi_summary(filtered_df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Employees", f"{kpis['total_employees']:,}")
        col2.metric("Attrition", f"{kpis['attrition_rate']:.1f}%")
        col3.metric("Avg. salary", f"${kpis['average_monthly_salary']:,.0f}")
        col4.metric("Avg. tenure", f"{kpis['average_tenure_years']:.1f} yrs")

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("Active", f"{kpis['active_employees']:,}")
        col6.metric("Job satisfaction", f"{kpis['average_job_satisfaction']:.1f}/4")
        col7.metric("Recent promotions", f"{kpis['recent_promotion_rate']:.1f}%")
        col8.metric("Overtime", f"{kpis['overtime_percentage']:.1f}%")

        st.divider()

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Employees who left vs. stayed")

            attrition_counts = (
                filtered_df["attrition_flag"]
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Employees")
            )

            attrition_counts["Status"] = attrition_counts["Status"].map(
                {0: "Stayed", 1: "Left"}
            )

            fig = px.bar(
                attrition_counts,
                x="Status",
                y="Employees",
                color="Status",
                color_discrete_map={"Stayed": "#0EA5E9", "Left": "#F43F5E"},
                text="Employees",
            )

            fig.update_layout(
                showlegend=False,
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig, use_container_width=True)

        with right_col:
            st.subheader("Attrition by department")

            department_attrition = (
                filtered_df.groupby("department")["attrition_flag"]
                .mean()
                .mul(100)
                .round(1)
                .sort_values(ascending=False)
                .rename("Attrition rate (%)")
                .reset_index()
            )

            fig2 = px.bar(
                department_attrition,
                x="department",
                y="Attrition rate (%)",
                color="Attrition rate (%)",
                color_continuous_scale=["#CCFBF1", "#0F766E", "#134E4A"],
                text_auto=True,
            )

            fig2.update_layout(
                showlegend=False,
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig2, use_container_width=True)


with risk_tab:
    st.subheader("Select an employee")

    if filtered_df.empty:
        st.warning("No employees match the selected filters.")
        st.stop()

    employee_options = filtered_df.index.tolist()

    selected_employee_id = st.selectbox(
        "Employee",
        employee_options,
        format_func=lambda employee_id: (
            f"{filtered_df.loc[employee_id, 'job_role']} — "
            f"{filtered_df.loc[employee_id, 'department']} — "
            f"Employee {employee_id}"
        ),
    )

    employee_df = filtered_df.loc[[selected_employee_id]]

    prediction = model.predict(employee_df)[0]
    prediction_proba = model.predict_proba(employee_df)[0, 1]
    risk_tier = get_risk_tier(prediction_proba)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Attrition probability", f"{prediction_proba:.0%}")

        if risk_tier == "High":
            st.error(f"**{risk_tier} risk**")
        elif risk_tier == "Medium":
            st.warning(f"**{risk_tier} risk**")
        else:
            st.success(f"**{risk_tier} risk**")

        st.progress(
            min(prediction_proba, 1.0),
            text=f"Model score: {prediction_proba:.1%}",
        )

    with col2:
        st.info(
            "This estimate is based on the selected employee’s HR profile. "
            "Use it to prioritize supportive conversations, not automated decisions."
        )

    st.divider()

    st.subheader("What influenced this prediction?")

    with st.spinner("Calculating explanation..."):
        shap_values, transformed_employee = compute_shap_values(
            model=model,
            X_background=df.head(200),
            X_explain=employee_df,
        )

    top_factors = get_top_factors_for_employee(
        shap_values=shap_values,
        transformed_features=transformed_employee,
        employee_position=0,
        top_n=8,
    )

    display_factors = top_factors[
        ["feature", "feature_value", "shap_value", "impact_direction"]
    ].copy()

    display_factors = display_factors.rename(
        columns={
            "feature": "Factor",
            "feature_value": "Value",
            "shap_value": "Impact",
            "impact_direction": "Effect",
        }
    )

    display_factors["Impact"] = display_factors["Impact"].round(4)

    display_factors["Effect"] = display_factors["Effect"].replace(
        {
            "increases_attrition_risk": "Increases risk",
            "decreases_attrition_risk": "Decreases risk",
        }
    )

    st.dataframe(display_factors, use_container_width=True, hide_index=True)

    st.subheader("Possible retention levers")

    recommendations = []

    if "overtime_flag" in filtered_df.columns:
        overtime_value = filtered_df.loc[selected_employee_id, "overtime_flag"]

        if overtime_value == 1:
            recommendations.append(
                "Review workload, staffing coverage, and flexible scheduling options."
            )

    if "job_satisfaction" in filtered_df.columns:
        satisfaction_value = filtered_df.loc[selected_employee_id, "job_satisfaction"]

        if satisfaction_value <= 2:
            recommendations.append(
                "Schedule a manager check-in to understand engagement and concerns."
            )

    if "years_since_promotion" in filtered_df.columns:
        promotion_value = filtered_df.loc[selected_employee_id, "years_since_promotion"]

        if promotion_value >= 3:
            recommendations.append(
                "Discuss career progression, skill development, and internal opportunities."
            )

    if not recommendations:
        recommendations.append(
            "Maintain regular manager check-ins and monitor changes in engagement."
        )

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    with st.expander("See detailed explanation"):
        ax = shap.plots.waterfall(
            shap_values[0],
            max_display=10,
            show=False,
        )

        st.pyplot(ax.figure)


with people_tab:
    st.subheader("Employees to review")

    if filtered_df.empty:
        st.warning("No employees match the selected filters.")
        st.stop()

    employee_scores = filtered_df.copy()
    employee_scores["attrition_probability"] = model.predict_proba(employee_scores)[:, 1]
    employee_scores["risk_tier"] = employee_scores["attrition_probability"].apply(get_risk_tier)

    review_columns = [
        "job_role",
        "department",
        "attrition_probability",
        "risk_tier",
        "monthly_income",
        "years_at_company",
        "job_satisfaction",
        "overtime_flag",
    ]

    risk_table = (
        employee_scores[review_columns]
        .sort_values("attrition_probability", ascending=False)
        .head(25)
        .copy()
    )

    risk_table["attrition_probability"] = risk_table["attrition_probability"].round(3)

    risk_table = risk_table.rename(
        columns={
            "job_role": "Job role",
            "department": "Department",
            "attrition_probability": "Attrition probability",
            "risk_tier": "Risk tier",
            "monthly_income": "Monthly income",
            "years_at_company": "Years at company",
            "job_satisfaction": "Job satisfaction",
            "overtime_flag": "Overtime",
        }
    )

    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    csv_data = risk_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download review list",
        data=csv_data,
        file_name="attrition_review_list.csv",
        mime="text/csv",
    )

    st.caption(
        "This list highlights employees with the highest model-estimated attrition probability among the selected filters."
    )


with segmentation_tab:
    st.subheader("Employee segments")

    st.caption(
        "Employees are grouped using KMeans clustering based on career, compensation, engagement, and workload-related features."
    )

    segmentation_X, segmentation_features_used = prepare_segmentation_features(
        df,
        features=segmentation_features,
    )

    segmentation_model.scaler = segmentation_scaler

    cluster_labels = assign_cluster(
        model=segmentation_model,
        df_new=segmentation_X,
        features=segmentation_features_used,
    )

    segmented_df = df.loc[filtered_df.index].copy()
    segmented_df["cluster"] = cluster_labels[df.index.isin(filtered_df.index)]

    if segmented_df.empty:
        st.warning("No employees match the selected filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Segment sizes")

            cluster_sizes = (
                segmented_df["cluster"]
                .value_counts()
                .sort_index()
                .rename_axis("Cluster")
                .reset_index(name="Employees")
            )

            cluster_sizes["Cluster"] = cluster_sizes["Cluster"].astype(str)

            fig3 = px.bar(
                cluster_sizes,
                x="Cluster",
                y="Employees",
                color="Employees",
                color_continuous_scale=["#99F6E4", "#0F766E", "#134E4A"],
                text_auto=True,
            )

            fig3.update_layout(
                showlegend=False,
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.markdown("#### Attrition by segment")

            cluster_attrition = (
                segmented_df.groupby("cluster")["attrition_flag"]
                .mean()
                .mul(100)
                .round(1)
                .sort_index()
                .rename("Attrition rate (%)")
                .reset_index()
            )

            cluster_attrition["cluster"] = cluster_attrition["cluster"].astype(str)

            fig4 = px.bar(
                cluster_attrition,
                x="cluster",
                y="Attrition rate (%)",
                color="Attrition rate (%)",
                color_continuous_scale=["#FDE68A", "#F59E0B", "#B45309"],
                text_auto=True,
            )

            fig4.update_layout(
                showlegend=False,
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        st.markdown("#### Segment profiles")

        profile_features = [
            feature
            for feature in segmentation_features_used
            if feature in segmented_df.columns
        ]

        cluster_profile = (
            segmented_df.groupby("cluster")[profile_features]
            .mean()
            .round(2)
        )

        cluster_profile.index = cluster_profile.index.astype(str)

        st.dataframe(cluster_profile, use_container_width=True)

        st.caption("Each row shows the average profile of employees in that segment.")

        st.divider()

        st.markdown("#### Sample employees")

        selected_cluster = st.selectbox(
            "Select segment",
            sorted(segmented_df["cluster"].unique().tolist()),
            format_func=lambda cluster: f"Cluster {cluster}",
        )

        sample_columns = [
            column
            for column in [
                "job_role",
                "department",
                "age",
                "monthly_income",
                "years_at_company",
                "job_satisfaction",
                "overtime_flag",
                "attrition_flag",
            ]
            if column in segmented_df.columns
        ]

        sample_employees = segmented_df[
            segmented_df["cluster"] == selected_cluster
        ][sample_columns].head(10)

        st.dataframe(sample_employees, use_container_width=True, hide_index=True)


with compensation_tab:
    st.subheader("Compensation Intelligence")
    st.caption(
        "Understand salary distribution, compare compensation across workforce groups, and estimate a market-aligned salary."
    )

    salary_col = "monthly_income"

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Employees", f"{len(filtered_df):,}")
    k2.metric("Average Salary", f"${filtered_df[salary_col].mean():,.0f}")
    k3.metric("Median Salary", f"${filtered_df[salary_col].median():,.0f}")
    k4.metric("Total Payroll", f"${filtered_df[salary_col].sum():,.0f}")

    st.write("")

    comp_tab1, comp_tab2, comp_tab3 = st.tabs(
        ["📊 Compensation overview", "⚖️ Pay equity", "🤖 Salary predictor"]
    )

    with comp_tab1:
        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            st.markdown("#### Salary distribution")
            st.caption("See how compensation is distributed across the selected workforce.")

            salary_fig = px.histogram(
                filtered_df,
                x=salary_col,
                nbins=40,
                color_discrete_sequence=["#0F766E"],
                labels={salary_col: "Monthly Salary"},
            )

            salary_fig.update_layout(
                height=420,
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(salary_fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### Department benchmark")
            st.caption("Compare average compensation across departments.")

            department_salary = (
                filtered_df.groupby("department")[salary_col]
                .mean()
                .reset_index()
                .sort_values(salary_col, ascending=False)
            )

            dept_fig = px.bar(
                department_salary,
                x=salary_col,
                y="department",
                orientation="h",
                color=salary_col,
                color_continuous_scale=["#CCFBF1", "#0F766E", "#134E4A"],
                labels={salary_col: "Average Monthly Salary", "department": "Department"},
            )

            dept_fig.update_layout(
                height=420,
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(dept_fig, use_container_width=True)

    with comp_tab2:
        st.markdown("#### Pay equity explorer")
        st.caption("Compare average salary across workforce dimensions to identify potential compensation gaps.")

        equity_options = [
            column
            for column in ["gender", "department", "job_role", "job_level"]
            if column in filtered_df.columns
        ]

        if equity_options:
            equity_col = st.selectbox(
                "Compare salary by",
                equity_options,
                format_func=lambda column: column.replace("_", " ").title(),
            )

            equity_df = (
                filtered_df.groupby(equity_col)[salary_col]
                .agg(["mean", "median", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )

            equity_chart_col, equity_table_col = st.columns([3, 2])

            with equity_chart_col:
                equity_fig = px.bar(
                    equity_df,
                    x=equity_col,
                    y="mean",
                    color="mean",
                    color_continuous_scale=["#FDE68A", "#F59E0B", "#0F766E"],
                    labels={
                        equity_col: equity_col.replace("_", " ").title(),
                        "mean": "Average Monthly Salary",
                    },
                )

                equity_fig.update_layout(
                    height=430,
                    showlegend=False,
                    margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(equity_fig, use_container_width=True)

            with equity_table_col:
                st.dataframe(
                    equity_df.rename(
                        columns={
                            "mean": "Average Salary",
                            "median": "Median Salary",
                            "count": "Employees",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No suitable compensation dimensions are available.")

    with comp_tab3:
        st.markdown("#### Estimate market-aligned salary")
        st.caption(
            "Enter an employee profile to generate a salary estimate. "
            "Final compensation should also consider performance, internal equity, and HR policy."
        )

        model_features = getattr(salary_model, "feature_names_in_", None)

        if not model_features:
            st.warning(
                "The salary model does not expose feature names, so a prediction form cannot be generated safely."
            )
        else:
            with st.form("salary_prediction_form"):
                st.markdown("##### Employee profile")

                input_columns = st.columns(3)
                salary_input = {}

                numeric_defaults = {
                    "age": 35,
                    "total_working_years": 10,
                    "years_at_company": 5,
                    "years_in_current_role": 3,
                    "years_since_promotion": 2,
                    "years_with_curr_manager": 4,
                    "num_companies_worked": 2,
                    "distance_from_home": 10,
                    "education": 3,
                    "job_level": 2,
                    "performance_rating": 3,
                }

                categorical_defaults = {
                    "department": "Research & Development",
                    "job_role": "Sales Executive",
                    "education_field": "Life Sciences",
                    "gender": "Male",
                    "marital_status": "Married",
                    "business_travel": "Travel_Rarely",
                    "over_time": "No",
                }

                for index, feature in enumerate(model_features):
                    column = input_columns[index % 3]

                    if feature in numeric_defaults:
                        salary_input[feature] = column.number_input(
                            feature.replace("_", " ").title(),
                            min_value=0.0,
                            value=float(numeric_defaults[feature]),
                            step=1.0,
                        )
                    elif feature in categorical_defaults:
                        default_value = categorical_defaults[feature]

                        if feature in df.columns:
                            options = sorted(
                                df[feature].dropna().astype(str).unique().tolist()
                            )
                        else:
                            options = [default_value]

                        if default_value not in options:
                            options.insert(0, default_value)

                        salary_input[feature] = column.selectbox(
                            feature.replace("_", " ").title(),
                            options,
                            index=options.index(default_value),
                        )
                    else:
                        salary_input[feature] = column.text_input(
                            feature.replace("_", " ").title(),
                            "",
                        )

                submitted = st.form_submit_button(
                    "Predict Salary",
                    use_container_width=True,
                )

            if submitted:
                try:
                    prediction_input = pd.DataFrame([salary_input])

                    for feature in model_features:
                        if feature in prediction_input.columns and feature in df.columns:
                            if pd.api.types.is_numeric_dtype(df[feature]):
                                prediction_input[feature] = pd.to_numeric(
                                    prediction_input[feature],
                                    errors="coerce",
                                )

                    salary_prediction = salary_model.predict(prediction_input)[0]
                    salary_prediction = max(float(salary_prediction), 0)

                    st.success(
                        f"### Predicted Monthly Salary: ${salary_prediction:,.0f}"
                    )

                    median_salary = filtered_df[salary_col].median()
                    salary_difference = salary_prediction - median_salary
                    salary_percent = (
                        (salary_difference / median_salary) * 100
                        if median_salary
                        else 0
                    )

                    result_col1, result_col2, result_col3 = st.columns(3)

                    result_col1.metric(
                        "Selected Workforce Median",
                        f"${median_salary:,.0f}",
                    )
                    result_col2.metric(
                        "Difference",
                        f"${salary_difference:,.0f}",
                    )
                    result_col3.metric(
                        "Vs Median",
                        f"{salary_percent:+.1f}%",
                    )

                    st.caption(
                        "This estimate is a decision-support reference and should be reviewed alongside market benchmarks and internal compensation policy."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        if salary_metrics:
            st.divider()
            st.markdown("#### Model quality")

            metric_columns = st.columns(len(salary_metrics))

            for column, (metric_name, metric_value) in zip(
                metric_columns, salary_metrics.items()
            ):
                if isinstance(metric_value, (int, float)):
                    column.metric(
                        metric_name.replace("_", " ").title(),
                        f"{metric_value:,.3f}",
                    )
                else:
                    column.metric(
                        metric_name.replace("_", " ").title(),
                        str(metric_value),
                    )
        else:
            st.info("Salary model metrics file was not found.")


with simulator_tab:
    st.subheader("Explore a hypothetical employee")

    st.caption(
        "Adjust key profile settings to see how the model estimates attrition risk. "
        "This is a scenario-analysis tool, not a guarantee of future outcomes."
    )

    col1, col2, col3 = st.columns(3)

    monthly_income = col1.slider(
        "Monthly income",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=100,
    )

    overtime_choice = col2.selectbox("Works overtime?", options=["No", "Yes"])
    job_satisfaction = col3.slider("Job satisfaction", min_value=1, max_value=4, value=3)

    col4, col5, col6 = st.columns(3)

    years_at_company = col4.slider("Years at company", min_value=0, max_value=40, value=5)
    years_since_promotion = col5.slider("Years since promotion", min_value=0, max_value=15, value=3)
    age = col6.slider("Age", min_value=18, max_value=60, value=35)

    sample_employee = df.iloc[[0]].copy()

    sample_employee["monthly_income"] = monthly_income
    sample_employee["overtime_flag"] = 1 if overtime_choice == "Yes" else 0
    sample_employee["job_satisfaction"] = job_satisfaction
    sample_employee["years_at_company"] = years_at_company
    sample_employee["years_since_promotion"] = years_since_promotion
    sample_employee["age"] = age

    simulator_probability = model.predict_proba(sample_employee)[0, 1]
    simulator_tier = get_risk_tier(simulator_probability)

    st.divider()

    result_col, explanation_col = st.columns([1, 2])

    with result_col:
        st.metric("Estimated attrition probability", f"{simulator_probability:.0%}")

        if simulator_tier == "High":
            st.error(f"**{simulator_tier} risk**")
        elif simulator_tier == "Medium":
            st.warning(f"**{simulator_tier} risk**")
        else:
            st.success(f"**{simulator_tier} risk**")

        st.progress(
            min(simulator_probability, 1.0),
            text=f"Model score: {simulator_probability:.1%}",
        )

    with explanation_col:
        st.info(
            "The simulator changes only the selected fields while keeping other employee attributes from the baseline profile unchanged."
        )

    with st.expander("How to use this responsibly"):
        st.markdown(
            """
            - Use the simulator to explore possible workforce scenarios, not to make decisions about a real individual.
            - Changing one input does not prove that changing it in real life would change attrition.
            - The model reflects historical patterns in the training data and may not generalize to every workforce.
            """
        )


with model_tab:
    st.subheader("Model quality")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    col4.metric("F1 score", f"{f1_score(y_test, y_pred):.3f}")

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Confusion matrix")

        matrix = confusion_matrix(y_test, y_pred)

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual: Stayed", "Actual: Left"],
            columns=["Predicted: Stayed", "Predicted: Left"],
        )

        st.dataframe(matrix_df, use_container_width=True)

    with right_col:
        st.subheader("How to interpret")

        st.markdown(
            """
            - **ROC-AUC** measures how well the model separates employees who leave from those who stay.
            - **Precision** measures how many high-risk predictions were actually correct.
            - **Recall** measures how many actual leavers the model identified.
            - **F1 score** balances precision and recall.
            """
        )

    st.divider()

    with st.expander("How to use this responsibly"):
        st.markdown(
            """
            - Use predictions to prioritize supportive conversations, not automated employment decisions.
            - Do not use this tool to punish, discipline, or make final decisions about individuals.
            - SHAP values explain the model’s reasoning, not proven causes of attrition.
            - Review model performance, data quality, and fairness regularly before operational use.
            """
        )