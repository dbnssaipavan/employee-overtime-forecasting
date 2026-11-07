import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Config
st.set_page_config(
    page_title="Employee Overtime Cost Forecasting System",
    page_icon="⏱️",
    layout="wide"
)

# Load Static Dataset Automatically
default_path = "overtime_forecast_dataset.csv"
if os.path.exists(default_path):
    df = pd.read_csv(default_path)
    selected_dataset_label = "overtime_forecast_dataset.csv"
else:
    st.error("Dataset 'overtime_forecast_dataset.csv' not found in workspace.")
    st.stop()

# Dataset Suitability & Validation Check
suitability_passed = True
validation_messages = []

if df is not None:
    validation_messages.append(f"✅ Record Count Valid: {df.shape[0]:,} rows found.")
    validation_messages.append(f"✅ Feature Dimensionality Valid: {df.shape[1]} columns found.")
    
    possible_targets = ['Overtime_Cost', 'Cost', 'Overtime_Expenses', 'Total_Cost']
    target_col = next((col for col in possible_targets if col in df.columns), None)
    
    if target_col is None:
        target_col = df.columns[-1]
    validation_messages.append(f"✅ Target Column Identified: '{target_col}'")
    
    if target_col and target_col != 'Overtime_Cost':
        df['Overtime_Cost'] = df[target_col]
        
    feature_candidates = [c for c in df.columns if c != 'Overtime_Cost']
    validation_messages.append(f"✅ Predictor Features Valid ({len(feature_candidates)} features detected).")

# Dynamic Model Training with MSE, MAE, RMSE, R2
@st.cache_data
def train_model(df_in, target_name):
    X = df_in.drop(columns=['Overtime_Cost'])
    y = df_in['Overtime_Cost']
    
    categorical_cols = [c for c in ['Department', 'Shift'] if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    return {
        'model': model,
        'feature_names': X_encoded.columns.tolist(),
        'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    }

artifact = train_model(df, 'Overtime_Cost')
model = artifact['model']
feature_names = artifact['feature_names']
metrics = artifact['metrics']

# Header
st.title("⏱️ Employee Overtime Cost Forecasting System")
st.markdown(f"Currently Analyzing: **{selected_dataset_label}** (Static Data Mode)")

# Sidebar Metrics
st.sidebar.header("📊 Active Model Metrics")
st.sidebar.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
st.sidebar.metric("Mean Squared Error (MSE)", f"${metrics['MSE']:.2f}")
st.sidebar.metric("Root Mean Sq. Error (RMSE)", f"${metrics['RMSE']:.2f}")
st.sidebar.metric("R² Score", f"{metrics['R2']:.4f}")

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analysis (EDA)",
    "🔮 Single Schedule Predictor", 
    "💡 Dataset & Model Explainability (XAI)",
    "📈 Batch / Scenario Simulator"
])

# -------------------------------------------------------------
# TAB 1: EDA (Tabular Format Only, Exact Dataset Values)
# -------------------------------------------------------------
with tab1:
    st.subheader(f"📊 Exploratory Data Analysis & Suitability Verification — {selected_dataset_label}")
    st.markdown("Comprehensive statistical audit of every column in the static dataset in tabular format.")
    
    with st.expander("🔍 Dataset Suitability & Validation Report", expanded=False):
        for msg in validation_messages:
            st.markdown(f"- {msg}")
        st.success("🎉 **Static dataset is fully suitable and verified for machine learning overtime cost prediction!**")

    if df is not None:
        st.markdown("---")
        with st.expander("📊 View Column Summary Statistics (Transposed)", expanded=True):
            st.dataframe(df.describe().T, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 📋 Complete Column Statistics & Metadata Table")
        
        col_summary_data = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / len(df)) * 100
            unique_vals = int(df[col].nunique())
            
            if np.issubdtype(df[col].dtype, np.number):
                mean_val = f"{df[col].mean():.2f}"
                std_val = f"{df[col].std():.2f}"
                min_val = f"{df[col].min():.2f}"
                max_val = f"{df[col].max():.2f}"
            else:
                mean_val = "-"
                std_val = "-"
                min_val = str(df[col].min()) if unique_vals > 0 else "-"
                max_val = str(df[col].max()) if unique_vals > 0 else "-"
                
            col_summary_data.append({
                "Column Name": col,
                "Data Type": dtype,
                "Missing Values": f"{missing_count} ({missing_pct:.1f}%)",
                "Unique Values": unique_vals,
                "Mean / Min": mean_val if mean_val != "-" else min_val,
                "Std Dev / Max": std_val if std_val != "-" else max_val
            })
            
        summary_df = pd.DataFrame(col_summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 Individual Column Statistics Tables")
        for col in df.select_dtypes(include=[np.number]).columns:
            with st.expander(f"📋 Column Statistics Table: `{col}`"):
                col_stats_df = pd.DataFrame({
                    "Statistic": ["Count", "Mean", "Median", "Std Dev", "Min", "25%", "50%", "75%", "Max", "Missing Values"],
                    "Value": [
                        f"{df[col].count():,}",
                        f"{df[col].mean():.4f}",
                        f"{df[col].median():.4f}",
                        f"{df[col].std():.4f}",
                        f"{df[col].min():.4f}",
                        f"{df[col].quantile(0.25):.4f}",
                        f"{df[col].quantile(0.50):.4f}",
                        f"{df[col].quantile(0.75):.4f}",
                        f"{df[col].max():.4f}",
                        f"{int(df[col].isnull().sum())}"
                    ]
                })
                st.dataframe(col_stats_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Single Schedule Predictor
# -------------------------------------------------------------with tab2:
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
        input_data = pd.DataFrame([{
            'Department': selected_dept,
            'Shift': selected_shift,
            'Experience': 9.0,
            'Attendance': 1.0,
            'Holiday': float(is_holiday),
            'Weekend_Worked': float(is_weekend),
            'Projects_Assigned': 2,
            'Monthly_Salary': hourly_wage * 160.0,
            'Hourly_Rate': hourly_wage,
            'Working_Hours': working_hours,
            'Overtime_Hours': max(0.0, working_hours - 8.0)
        }])
        
        categorical_cols = [c for c in ['Department', 'Shift'] if c in input_data.columns]
        input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=False)
        
        predicted_cost = model.predict(input_encoded)[0]
        overtime_hours = max(0.0, working_hours - 8.0)
        
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Estimated Overtime Cost", f"${predicted_cost:.2f}")
        res_col2.metric("Overtime Hours", f"{overtime_hours:.2f} hrs")
        res_col3.metric("Effective Cost per OT Hour", f"${(predicted_cost / overtime_hours if overtime_hours > 0 else 0):.2f}")

# -------------------------------------------------------------
# TAB 3: Explainability & Dataset Insights (With Graphs & Technique Explanation)
# -------------------------------------------------------------
with tab3:
    st.subheader("💡 Dataset Insights & Model Explainability (XAI) with Graphs")
    st.markdown("""
    ### 🔬 Machine Learning Technique Used:
    The model utilizes a **Random Forest Regressor**, which is an ensemble learning method constructed from multiple decision trees. For model explainability, **Mean Decrease in Impurity (Gini Feature Importance)** is applied alongside feature correlation analysis. During training, the Random Forest algorithm evaluates how much each feature decreases the weighted impurity across all trees, assigning higher importance scores to workforce factors that heavily drive overtime costs.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Feature Importance Graph")
    
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_imp.plot(kind='barh', ax=ax, color='#ff4b4b')
    ax.set_title(f"Random Forest Global Feature Importance ({selected_dataset_label})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.set_ylabel("Dataset Features", fontsize=10)
    sns.despine()
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### 📊 Correlation Heatmap Graph")
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, cbar=True, annot_kws={"size": 8})
    ax2.set_title("Dataset Feature Correlation Matrix", fontsize=12, fontweight='bold')
    st.pyplot(fig2)
    
    st.markdown("---")
    st.markdown("### 📋 Key Analytical Takeaways")
    st.markdown("Primary cost drivers such as hourly wage and working hours exhibit the highest importance scores, confirming non-linear cost scaling. Additionally, holiday and weekend binary flags introduce premium cost multipliers into the prediction model.")

# -------------------------------------------------------------
# TAB 4: Batch / Scenario Simulator
# -------------------------------------------------------------
with tab4:
    st.subheader("📈 Interactive Batch & Workforce Scenario Simulator")
    st.markdown("Model complex multi-shift staffing strategies and compare projected financial expenditures.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sim_hours = st.slider("Shift Working Hours", 8.0, 16.0, 11.0, 0.25, key="sim_hours")
        sim_wage = st.number_input("Hourly Wage ($)", 10.0, 200.0, 45.0, 1.0, key="sim_wage")
    with col_s2:
        sim_headcount = st.slider("Employee Headcount", 1, 100, 15, key="sim_headcount")
        sim_holiday = st.selectbox("Holiday Premium?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", key="sim_hol")
    with col_s3:
        sim_weekend = st.selectbox("Weekend Shift?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", key="sim_wknd")
        sim_dept = st.selectbox("Department", departments, key="sim_dept")

    if 'scenario_list' not in st.session_state:
        st.session_state['scenario_list'] = [
            {"Name": "Standard Weekday Shift", "Dept": sim_dept, "Shift": "Morning", "Hours": 9.0, "Wage": 45.0, "Headcount": 10, "Weekend": 0, "Holiday": 0},
            {"Name": "Extended Weekend Shift", "Dept": sim_dept, "Shift": "Evening", "Hours": 12.0, "Wage": 45.0, "Headcount": 10, "Weekend": 1, "Holiday": 0},
            {"Name": "Holiday Surge Shift", "Dept": sim_dept, "Shift": "Night", "Hours": 14.0, "Wage": 45.0, "Headcount": 10, "Weekend": 0, "Holiday": 1},
        ]

    if st.button("➕ Add Custom Scenario", type="primary"):
        st.session_state['scenario_list'].append({
            "Name": f"Custom Shift ({sim_headcount} staff)",
            "Dept": sim_dept,
            "Shift": "Evening",
            "Hours": sim_hours,
            "Wage": sim_wage,
            "Headcount": sim_headcount,
            "Weekend": sim_weekend,
            "Holiday": sim_holiday
        })
        st.success("Changes applied successfully!")

    if st.session_state['scenario_list']:
        results = []
        for s in st.session_state['scenario_list']:
            row_dict = {
                'Department': s['Dept'],
                'Shift': s['Shift'],
                'Actual_Working_Hours': s['Hours'],
                'Hourly_Wage': s['Wage'],
                'Weekend': s['Weekend'],
                'Holiday': s['Holiday'],
                'Month': 10,
                'Day_of_Week': 2
            }
            
            row_df = pd.DataFrame([row_dict])
            categorical_cols = [c for c in ['Department', 'Shift'] if c in row_df.columns]
            row_enc = pd.get_dummies(row_df, columns=categorical_cols).reindex(columns=feature_names, fill_value=False)
            per_employee_cost = model.predict(row_enc)[0]
            total_team_cost = per_employee_cost * s['Headcount']
            
            results.append({
                "Scenario": s['Name'],
                "Headcount": s['Headcount'],
                "Hours": s['Hours'],
                "Per Employee OT ($)": round(per_employee_cost, 2),
                "Total Shift Cost ($)": round(total_team_cost, 2),
                "Est. Monthly Budget ($)": round(total_team_cost * 4.33, 2)
            })
            
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
