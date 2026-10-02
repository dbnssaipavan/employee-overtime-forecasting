import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Overtime Cost Forecasting System",
    page_icon="⏱️",
    layout="wide"
)

# Load Saved Artifacts
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

try:
    artifact = load_model()
    model = artifact['model']
    feature_names = artifact['feature_names']
    departments = artifact['departments']
    shifts = artifact['shifts']
    metrics = artifact['metrics']
except Exception as e:
    st.error("Error loading 'rf_model.pkl'. Please run 'python train_model.py' first.")
    st.stop()

# Header
st.title("⏱️ Employee Overtime Cost Forecasting System")
st.markdown("Forecast overtime costs to support workforce budgeting and scheduling optimization.")

st.sidebar.header("📊 Model Metrics")
st.sidebar.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
st.sidebar.metric("Root Mean Sq. Error (RMSE)", f"${metrics['RMSE']:.2f}")
st.sidebar.metric("R² Score", f"{metrics['R2']:.4f}")

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Single Schedule Predictor", "📈 Batch / Scenario Simulator", "💡 Model Explainability (XAI)"])

with tab1:
    st.subheader("Input Workforce Schedule Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_dept = st.selectbox("Department", departments)
        selected_shift = st.selectbox("Shift", shifts)
        working_hours = st.slider("Total Working Hours", min_value=8.0, max_value=16.0, value=11.0, step=0.25)
        
    with col2:
        hourly_wage = st.number_input("Hourly Wage ($)", min_value=10.0, max_value=200.0, value=45.0, step=1.0)
        month = st.selectbox("Month", list(range(1, 13)), index=9) # Default October
        
    with col3:
        day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        day_of_week = st.selectbox("Day of Week", list(day_map.keys()), format_func=lambda x: day_map[x])
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_holiday = st.selectbox("Holiday Shift?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        st.info(f"Weekend Status: **{'Yes' if is_weekend == 1 else 'No'}**")

    if st.button("Calculate Overtime Forecast", type="primary"):
        # Format input dataframe
        input_data = pd.DataFrame([{
            'Department': selected_dept,
            'Shift': selected_shift,
            'Actual_Working_Hours': working_hours,
            'Hourly_Wage': hourly_wage,
            'Weekend': is_weekend,
            'Holiday': is_holiday,
            'Month': month,
            'Day_of_Week': day_of_week
        }])
        
        # Apply encoding matching training features
        input_encoded = pd.get_dummies(input_data, columns=['Department', 'Shift'])
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=False)
        
        # Predict
        predicted_cost = model.predict(input_encoded)[0]
        overtime_hours = max(0.0, working_hours - 8.0)
        
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Estimated Overtime Cost", f"${predicted_cost:.2f}")
        res_col2.metric("Overtime Hours", f"{overtime_hours:.2f} hrs")
        res_col3.metric("Effective Cost per OT Hour", f"${(predicted_cost / overtime_hours if overtime_hours > 0 else 0):.2f}")

with tab2:
    st.subheader("Compare Shift Schedules")
    st.write("Evaluate how adjusting working hours or shifting work to weekdays impacts total overtime expenses.")
    
    # Sample scenario table
    num_employees = st.slider("Number of Employees on Shift", 1, 50, 10)
    
    scenario_data = pd.DataFrame([
        {"Scenario": "Standard Weekday Shift", "Dept": selected_dept, "Shift": selected_shift, "Hours": 10.0, "Wage": hourly_wage, "Weekend": 0, "Holiday": 0, "Month": month, "Day": 1},
        {"Scenario": "Extended Weekend Shift", "Dept": selected_dept, "Shift": selected_shift, "Hours": 12.0, "Wage": hourly_wage, "Weekend": 1, "Holiday": 0, "Month": month, "Day": 5},
        {"Scenario": "Holiday Peak Workload", "Dept": selected_dept, "Shift": selected_shift, "Hours": 13.0, "Wage": hourly_wage, "Weekend": 0, "Holiday": 1, "Month": month, "Day": 2},
    ])
    
    preds = []
    for _, row in scenario_data.iterrows():
        row_df = pd.DataFrame([{
            'Department': row['Dept'], 'Shift': row['Shift'],
            'Actual_Working_Hours': row['Hours'], 'Hourly_Wage': row['Wage'],
            'Weekend': row['Weekend'], 'Holiday': row['Holiday'],
            'Month': row['Month'], 'Day_of_Week': row['Day']
        }])
        row_enc = pd.get_dummies(row_df, columns=['Department', 'Shift']).reindex(columns=feature_names, fill_value=False)
        single_cost = model.predict(row_enc)[0]
        preds.append(round(single_cost * num_employees, 2))
        
    scenario_data["Total Team Cost ($)"] = preds
    st.dataframe(scenario_data[["Scenario", "Hours", "Weekend", "Holiday", "Total Team Cost ($)"]], use_container_width=True)

with tab3:
    st.subheader("Explainable AI (XAI) & Key Cost Drivers")
    st.write("Feature importances derived from the Random Forest regression model:")
    
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_imp.tail(10).plot(kind='barh', ax=ax, color='#1f77b4')
    ax.set_title("Top 10 Drivers of Overtime Cost")
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)
    
    st.info("💡 **Cost Insight:** Working Hours, Hourly Wage, and Weekend/Holiday multipliers are the top drivers influencing overtime expenses.")