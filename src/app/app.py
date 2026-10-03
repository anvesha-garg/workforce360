import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
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
        .stApp {
            background-color: #fbfdfc;
        }

        section[data-testid="stSidebar"] {
            background-color: #eef6f4;
            border-right: 1px solid #d1e7e3;
        }

        h1, h2, h3 {
            color: #0f766e;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #d1e7e3;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(15, 118, 110, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            color: #4b5563;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #0f766e;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 8px 16px;
            border: 1px solid #d1e7e3;
        }

        .stTabs [aria-selected="true"] {
            background-color: #0f766e;
            color: white;
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


def get_risk_tier(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


df = load_clean_data()
model = load_model()
X_train, X_test, y_train, y_test = load_evaluation_data()


@st.cache_resource
def load_segmentation_artifact():
    return load_segmentation_model()


segmentation_artifact = load_segmentation_artifact()
segmentation_model = segmentation_artifact["model"]
segmentation_scaler = segmentation_artifact["scaler"]
segmentation_features = segmentation_artifact["features"]

with st.sidebar:
    st.header("Filters")

    departments = ["All departments"] + sorted(df["department"].unique().tolist())
    selected_department = st.selectbox("Department", departments)

    job_roles = ["All job roles"] + sorted(df["job_role"].unique().tolist())
    selected_job_role = st.selectbox("Job role", job_roles)

filtered_df = df.copy()

if selected_department != "All departments":
    filtered_df = filtered_df[
        filtered_df["department"] == selected_department
    ]

if selected_job_role != "All job roles":
    filtered_df = filtered_df[
        filtered_df["job_role"] == selected_job_role
    ]

st.title("Workforce360")
st.caption("Understand workforce health, attrition risk, and what drives it.")

overview_tab, risk_tab, people_tab, segmentation_tab, simulator_tab, model_tab = st.tabs(
    [
        "Workforce overview",
        "Attrition risk",
        "People to review",
        "Segmentation",
        "What-if simulator",
        "Model quality",
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
                {
                    0: "Stayed",
                    1: "Left",
                }
            )

            st.bar_chart(
                attrition_counts.set_index("Status"),
                use_container_width=True,
            )

        with right_col:
            st.subheader("Attrition by department")

            department_attrition = (
                filtered_df.groupby("department")["attrition_flag"]
                .mean()
                .mul(100)
                .round(1)
                .sort_values(ascending=False)
                .rename("Attrition rate (%)")
            )

            st.bar_chart(department_attrition, use_container_width=True)

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
        st.metric(
            "Attrition probability",
            f"{prediction_proba:.0%}",
        )

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
        [
            "feature",
            "feature_value",
            "shap_value",
            "impact_direction",
        ]
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

    st.dataframe(
        display_factors,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Possible retention levers")

    recommendations = []

    if "overtime_flag" in filtered_df.columns:
        overtime_value = filtered_df.loc[selected_employee_id, "overtime_flag"]

        if overtime_value == 1:
            recommendations.append(
                "Review workload, staffing coverage, and flexible scheduling options."
            )

    if "job_satisfaction" in filtered_df.columns:
        satisfaction_value = filtered_df.loc[
            selected_employee_id, "job_satisfaction"
        ]

        if satisfaction_value <= 2:
            recommendations.append(
                "Schedule a manager check-in to understand engagement and concerns."
            )

    if "years_since_promotion" in filtered_df.columns:
        promotion_value = filtered_df.loc[ selected_employee_id, "years_since_promotion"]

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
    employee_scores["attrition_probability"] = model.predict_proba(
        employee_scores
    )[:, 1]
    employee_scores["risk_tier"] = employee_scores[
        "attrition_probability"
    ].apply(get_risk_tier)

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

    risk_table["attrition_probability"] = risk_table[
        "attrition_probability"
    ].round(3)

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

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = risk_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download review list",
        data=csv_data,
        file_name="attrition_review_list.csv",
        mime="text/csv",
    )

    st.caption(
        "This list highlights employees with the highest model-estimated attrition "
        "probability among the selected filters."
    )

with model_tab:
    st.subheader("Model quality")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ROC-AUC",
        f"{roc_auc_score(y_test, y_proba):.3f}",
    )

    col2.metric(
        "Precision",
        f"{precision_score(y_test, y_pred):.3f}",
    )

    col3.metric(
        "Recall",
        f"{recall_score(y_test, y_pred):.3f}",
    )

    col4.metric(
        "F1 score",
        f"{f1_score(y_test, y_pred):.3f}",
    )

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

with segmentation_tab:
    st.subheader("Employee segments")

    st.caption(
        "Employees are grouped using KMeans clustering based on career, compensation, "
        "engagement, and workload-related features."
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

    segmented_df = filtered_df.copy()
    segmented_df["cluster"] = cluster_labels[
        filtered_df.index.isin(df.index)
    ]

    segmented_df = df.loc[filtered_df.index].copy()
    segmented_df["cluster"] = cluster_labels[
        df.index.isin(filtered_df.index)
    ]

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

            st.bar_chart(
                cluster_sizes.set_index("Cluster"),
                use_container_width=True,
            )

        with col2:
            st.markdown("#### Attrition by segment")

            cluster_attrition = (
                segmented_df.groupby("cluster")["attrition_flag"]
                .mean()
                .mul(100)
                .round(1)
                .sort_index()
                .rename("Attrition rate (%)")
            )

            cluster_attrition.index = cluster_attrition.index.astype(str)

            st.bar_chart(cluster_attrition, use_container_width=True)

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

        st.dataframe(
            cluster_profile,
            use_container_width=True,
        )

        st.caption(
            "Each row shows the average profile of employees in that segment."
        )

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

        st.dataframe(
            sample_employees,
            use_container_width=True,
            hide_index=True,
        )

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

    overtime_choice = col2.selectbox(
        "Works overtime?",
        options=["No", "Yes"],
    )

    job_satisfaction = col3.slider(
        "Job satisfaction",
        min_value=1,
        max_value=4,
        value=3,
    )

    col4, col5, col6 = st.columns(3)

    years_at_company = col4.slider(
        "Years at company",
        min_value=0,
        max_value=40,
        value=5,
    )

    years_since_promotion = col5.slider(
        "Years since promotion",
        min_value=0,
        max_value=15,
        value=3,
    )

    age = col6.slider(
        "Age",
        min_value=18,
        max_value=60,
        value=35,
    )

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
        st.metric(
            "Estimated attrition probability",
            f"{simulator_probability:.0%}",
        )

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
            "The simulator changes only the selected fields while keeping other "
            "employee attributes from the baseline profile unchanged."
        )

    with st.expander("How to use this responsibly"):
        st.markdown(
            """
            - Use the simulator to explore possible workforce scenarios, not to make decisions about a real individual.
            - Changing one input does not prove that changing it in real life would change attrition.
            - The model reflects historical patterns in the training data and may not generalize to every workforce.
            """
        )