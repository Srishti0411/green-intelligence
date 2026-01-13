import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import plotly.express as px

from models.regression import (
    run_regression_pipeline,
    load_data,
    prepare_features,
    train_model
)


# APP CONFIG

st.set_page_config(
    page_title="Green Intelligence",
    layout="wide"
)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
    }

    div[role="radiogroup"] {
        margin-bottom: 16px !important;
    }

    h1 {
        margin-top: 0.2em !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# NAVBAR

st.markdown(
    """
    <style>
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 16px;
        background: #020617;
        padding: 14px 20px;
        border-radius: 999px;
        margin-bottom: 32px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }

    div[role="radiogroup"] input {
        display: none;
    }

    div[role="radiogroup"] label {
        padding: 10px 18px;
        border-radius: 999px;
        font-size: 15px;
        font-weight: 500;
        color: #94a3b8;
        cursor: pointer;
        transition: all 0.25s ease;
        white-space: nowrap;
    }

    div[role="radiogroup"] label:hover {
        background: #0f172a;
        color: #e5e7eb;
    }
 
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #22c55e, #4ade80);
        color: #022c22;
        font-weight: 600;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.35);
    }
    </style>
    """,
    unsafe_allow_html=True
)

page = st.radio(
    "",
    [
        "Dashboard",
        "Regression Analysis",
        "Predictor",
        "Sustainable AI Insights"
    ],
    horizontal=True
)

#data load

DATA_PATH = "data/merged_dataset.csv"
df = pd.read_csv(DATA_PATH)


#DASHBOARD (HOME MERGED)
if page == "Dashboard":
    st.title("Green Intelligence")
    st.markdown(
        """
        This dashboard provides **overview** of the
        **energy consumption and carbon footprint** of machine learning
        workloads across different hardware configurations.
        """
    )

    # ---------- FILTERS ----------
    st.sidebar.header("Filter Data")

    selected_devices = st.sidebar.multiselect(
        "Select Device",
        df["device_name"].unique(),
        default=df["device_name"].unique()
    )

    selected_models = st.sidebar.multiselect(
        "Select Model Type",
        df["model_type"].unique(),
        default=df["model_type"].unique()
    )

    filtered_df = df[
        (df["device_name"].isin(selected_devices)) &
        (df["model_type"].isin(selected_models))
    ]

    # ---------- KPI CARDS ----------
    st.markdown("### Key Metrics")

    col1, col2, col3 = st.columns(3)

    avg_energy = filtered_df["energy_kwh"].mean()
    avg_co2 = filtered_df["co2_kg"].mean()

    most_efficient = (
        filtered_df.groupby("device_name")["energy_kwh"]
        .mean()
        .idxmin()
    )

    col1.metric("Avg Energy (kWh)", f"{avg_energy:.2f}")
    col2.metric("Avg CO₂ (kg)", f"{avg_co2:.2f}")
    col3.metric("Most Efficient Device", most_efficient)

    # ---------- CHARTS ----------
    st.markdown("### ⏱  Energy vs Runtime")

    fig1 = px.scatter(
        filtered_df,
        x="runtime_hours",
        y="energy_kwh",
        color="device_name",
        size="batch_size",
        hover_data=["model_type"],
        title="Energy Consumption vs Runtime"
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        fig2 = px.box(
            filtered_df,
            x="device_name",
            y="energy_kwh",
            title="Energy Distribution Across Hardware"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        fig3 = px.bar(
            filtered_df.groupby("device_name", as_index=False)["co2_kg"].mean(),
            x="device_name",
            y="co2_kg",
            title="Average CO₂ Emissions per Device"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # st.success(
    #     """
    #     💡 **Key Takeaway**

    #     Hardware choice has a **major impact** on energy usage and CO₂ emissions.
    #     Optimizing runtime and selecting efficient devices can significantly
    #     reduce the environmental footprint of ML workloads.
    #     """
    # )


# REGRESSION ANALYSIS
elif page == "Regression Analysis":
    st.title("Regression Analysis")

    st.markdown(
        """
        This section evaluates how well our **Linear Regression model**
        predicts energy consumption for machine learning workloads.
        """
    )

    results = run_regression_pipeline(DATA_PATH)

    metrics = results["metrics"]
    coefficients = results["coefficients"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]

    st.subheader("Dataset Overview")
    st.write(f"Total samples: **{len(df)}**")
    st.dataframe(df.head())

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", round(metrics["R2"], 3))
    col2.metric("RMSE", round(metrics["RMSE"], 3))
    col3.metric("MSE", round(metrics["MSE"], 3))

    st.subheader("Actual vs Predicted Energy")

    comparison_df = pd.DataFrame({
        "Actual Energy (kWh)": y_test,
        "Predicted Energy (kWh)": y_pred
    })

    fig4 = px.scatter(
        comparison_df,
        x="Actual Energy (kWh)",
        y="Predicted Energy (kWh)",
        title="Actual vs Predicted Energy Consumption",
        opacity=0.7
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Residual Analysis")

    residual_df = pd.DataFrame({
        "Predicted Energy (kWh)": y_pred,
        "Residual": y_test - y_pred
    })

    fig5 = px.scatter(
        residual_df,
        x="Predicted Energy (kWh)",
        y="Residual",
        title="Residuals vs Predicted Energy"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Feature Influence")

    fig6 = px.bar(
        coefficients,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Regression Coefficients"
    )
    st.plotly_chart(fig6, use_container_width=True)


#PREDICTOR PAGE
elif page == "Predictor":
    st.title("Energy & CO₂ Predictor")

    st.markdown(
        """
        Estimate the **energy consumption and carbon footprint**
        of a machine learning workload based on custom inputs.
        """
    )

    X, y = prepare_features(df)
    model, _, _, _ = train_model(X, y)

    st.markdown("### Configure Your ML Workload")

    col1, col2 = st.columns(2)

    with col1:
        device = st.selectbox("Device", df["device_name"].unique())
        model_type = st.selectbox("Model Type", df["model_type"].unique())

    with col2:
        runtime = st.slider("Runtime (hours)", 0.1, 48.0, 5.0)
        batch_size = st.selectbox("Batch Size", sorted(df["batch_size"].unique()))

    input_df = pd.DataFrame([{
        "device_name": device,
        "model_type": model_type,
        "runtime_hours": runtime,
        "batch_size": batch_size
    }])

    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    predicted_energy = model.predict(input_encoded)[0]

    EMISSION_FACTOR = 0.475  # kg CO2 per kWh
    predicted_co2 = predicted_energy * EMISSION_FACTOR

    st.markdown("### Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Energy (kWh)", f"{predicted_energy:.2f}")
    col2.metric("Estimated CO₂ (kg)", f"{predicted_co2:.2f}")

    # st.info(
    #     f"""
    #     💡 **Interpretation**

    #     Running a **{model_type}** model on **{device}**
    #     for **{runtime:.1f} hours** with batch size **{batch_size}**
    #     is estimated to consume **{predicted_energy:.2f} kWh**
    #     and emit approximately **{predicted_co2:.2f} kg of CO₂**.
    #     """
    # )

# SUSTAINABLE AI INSIGHTS
elif page == "Sustainable AI Insights":
    
    from ai_insights import generate_ai_insights

    st.title("AI-Powered Sustainability Insights")

    st.markdown(
        """
        Get **AI-generated recommendations** to reduce the energy consumption
        and carbon footprint of your machine learning workloads.

        Insights are generated based on **your system behavior and workload patterns**.
        """
    )

    st.divider()

    st.markdown("### Analysis Context")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Optimization Goals")

        goal_energy = st.checkbox("Reduce Energy Consumption")
        goal_carbon = st.checkbox("Reduce Carbon Emissions")
        goal_balance = st.checkbox("Balance Performance & Sustainability")


    with col2:
        hardware_scope = st.selectbox(
            "Hardware Scope",
            [
                "All Devices",
                "CPUs Only",
                "GPUs Only"
            ]
        )

    with col3:
        model_type = st.selectbox(
            "Model Type",
            [
                "All Models",
                "CNN",
                "Transformer",
                "LLM"
            ]
        )

    st.divider()

    generate = st.button("Generate AI Insights")

    selected_goals = []

    if goal_energy:
        selected_goals.append("Reduce Energy Consumption")
    if goal_carbon:
        selected_goals.append("Reduce Carbon Emissions")
    if goal_balance:
        selected_goals.append("Balance Performance & Sustainability")

    

   
    # AI Output (Groq)

    if generate:
        if not selected_goals:
            st.warning("Please select at least one optimization goal.")
            st.stop()
        with st.spinner("Generating sustainability insights..."):

            for goal in selected_goals:

                context = {
                    "goal": goal,
                    "hardware": hardware_scope,
                    "model": model_type,
                    "avg_energy": round(0.98, 2),
                    "avg_co2": round(0.47, 2),
                    "efficient_device": "Ryzen 9"
                }

                insights = generate_ai_insights(context)
                GOAL_TITLES = {
                    "Reduce Energy Consumption": " Energy Optimization Insights",
                    "Reduce Carbon Emissions": " Carbon Reduction Insights",
                    "Balance Performance & Sustainability": " Performance–Sustainability Trade-offs"
                }

                # Sectioned output
                st.markdown(f"### {GOAL_TITLES[goal]}")
                st.markdown(insights)
                st.divider()
