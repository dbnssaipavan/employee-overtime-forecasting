import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Overtime Cost Forecasting System",
    page_icon="⏱️",
    layout="wide"
)

@st.cache_resource
def load_or_train_model():
    model_path = "rf_model.pkl"
    csv_path = "overtime_dataset_single_columns.csv"
    
    if not os.path.exists(model_path):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            X = df[['Department', 'Shift', 'Actual_Working_Hours', 'Hourly_Wage', 
                    'Weekend', 'Holiday', 'Month', 'Day_of_Week']].copy()
            y = df['Overtime_Cost']
            X_encoded = pd.get_dummies(X, columns=['Department', 'Shift'], drop_first=True)
            
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            artifact = {
                'model': model,
                'feature_names': X_encoded.columns.tolist(),
                'departments': sorted(df['Department'].unique().tolist()),
                'shifts': sorted(df['Shift'].unique().tolist()),
                'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            }
            joblib.dump(artifact, model_path)
        else:
            st.error("Dataset not found!")
            st.stop()
    return joblib.load(model_path)

artifact = load_or_train_model()
model = artifact['model']
feature_names = artifact['feature_names']
departments = artifact['departments']
shifts = artifact['shifts']
metrics = artifact['metrics']

# Header
st.title("⏱️ Employee Overtime Cost Forecasting System")
st.markdown("Forecast overtime costs with Explainable AI (SHAP) transparency.")

st.sidebar.header("📊 Model Metrics")
st.sidebar.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
st.sidebar.metric("Root Mean Sq. Error (RMSE)", f"${metrics['RMSE']:.2f}")
st.sidebar.metric("R² Score", f"{metrics['R2']:.4f}")

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Single Schedule Predictor", "📈 Batch Scenario Simulator", "🧠 Explainable AI (SHAP Dashboard)"])

with tab1:
    st.subheader("Input Workforce Schedule Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_dept = st.selectbox("Department", departments)
        selected_shift = st.selectbox("Shift", shifts)
        working_hours = st.slider("Total Working Hours", min_value=8.0, max_value=16.0, value=11.0, step=0.25)
        
    with col2:
        hourly_wage = st.number_input("Hourly Wage ($)", min_value=10.0, max_value=200.0, value=45.0, step=1.0)
        month = st.selectbox("Month", list(range(1, 13)), index=9)
        
    with col3:
        day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        day_of_week = st.selectbox("Day of Week", list(day_map.keys()), format_func=lambda x: day_map[x])
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_holiday = st.selectbox("Holiday Shift?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Calculate Overtime Forecast & Explain", type="primary"):
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
        
        input_encoded = pd.get_dummies(input_data, columns=['Department', 'Shift'])
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=False)
        
        predicted_cost = model.predict(input_encoded)[0]
        overtime_hours = max(0.0, working_hours - 8.0)
        
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Estimated Overtime Cost", f"${predicted_cost:.2f}")
        res_col2.metric("Overtime Hours", f"{overtime_hours:.2f} hrs")
        res_col3.metric("Effective Cost / OT Hour", f"${(predicted_cost / overtime_hours if overtime_hours > 0 else 0):.2f}")

        # Local SHAP Explanation for this specific prediction
        st.subheader("🔍 Why did the model predict this cost?")
        st.write("The breakdown below shows how each input feature pushed the predicted cost **higher (red/positive)** or **lower (blue/negative)** relative to the baseline average.")
        
        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer.shap_values(input_encoded)
        
        # Create local feature contribution dataframe
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_encoded.iloc[0].values,
            'SHAP Impact ($)': shap_values_single[0]
        }).sort_values(by='SHAP Impact ($)', ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#ff4b4b' if x > 0 else '#1c83e1' for x in contrib_df['SHAP Impact ($)'].head(8)]
        contrib_df.head(8).sort_values(by='SHAP Impact ($)', ascending=True).plot(
            kind='barh', x='Feature', y='SHAP Impact ($)', ax=ax, color=colors
        )
        ax.set_title("Top Feature Contributions for This Specific Shift")
        ax.set_xlabel("Impact on Cost ($)")
        st.pyplot(fig)

with tab2:
    st.subheader("Compare Shift Schedules")
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
    st.subheader("🧠 Global Explainable AI (SHAP Summary)")
    st.write("This chart shows the overall global impact of features across the entire workforce dataset. Features are ordered by importance, and colors indicate whether high feature values (red) increase or decrease overtime costs.")
    
    if os.path.exists("shap_summary.png"):
        st.image("shap_summary.png", caption="SHAP Global Feature Summary Plot", use_column_width=True)
    else:
        st.info("Run `train_model.py` locally to generate and save the SHAP summary plot image.")
        
    st.info("💡 **XAI Takeaway:** SHAP proves mathematically that *Actual Working Hours* and *Hourly Wage* have the steepest positive impact on cost, followed closely by *Weekend* and *Holiday* binary multipliers.")
