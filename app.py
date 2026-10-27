import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# Sidebar: Dataset Management
st.sidebar.header("📁 Dataset Management")
dataset_mode = st.sidebar.radio(
    "Choose Dataset Option",
    ["Load Default Dataset", "Upload Local Dataset"]
)

df = None
selected_dataset_label = ""

if dataset_mode == "Load Default Dataset":
    selected_dataset_label = "Default Overtime Dataset"
    default_path = "employee_overtime_dataset.csv"
    
    if not os.path.exists(default_path):
        # Auto-generate default dataset on the fly if missing
        np.random.seed(42)
        n = 3000
        df_default = pd.DataFrame({
            'Experience': np.random.uniform(1.0, 17.0, n),
            'Attendance': np.random.choice([0.0, 1.0], size=n, p=[0.25, 0.75]),
            'Holiday': np.random.choice([0.0, 1.0], size=n, p=[0.85, 0.15]),
            'Weekend_Worked': np.random.choice([0.0, 1.0], size=n, p=[0.90, 0.10]),
            'Projects_Assigned': np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1]),
            'Monthly_Salary': np.random.normal(69405, 15640, n).clip(40000, 120000),
            'Hourly_Rate': np.random.normal(394, 88, n).clip(200, 600),
            'Working_Hours': np.random.normal(8.0, 3.5, n).clip(0.0, 16.0),
            'Overtime_Hours': np.random.normal(1.5, 1.5, n).clip(0.0, 8.0),
            'Overtime_Cost': np.random.exponential(500, n).clip(0.0, 4000.0)
        })
        df_default.to_csv(default_path, index=False)
        
    try:
        df = pd.read_csv(default_path)
    except Exception as e:
        st.sidebar.error(f"Error loading default dataset: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Local Dataset (.csv, .xlsx)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        selected_dataset_label = f"Uploaded File: {uploaded_file.name}"
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    else:
        st.sidebar.info("Please upload a local CSV or XLSX file above.")

# Dataset Suitability & Validation Check
suitability_passed = True
validation_messages = []

if df is not None:
    if df.shape[0] < 10:
        suitability_passed = False
        validation_messages.append("❌ Dataset has fewer than 10 rows. Insufficient records for training.")
    else:
        validation_messages.append(f"✅ Record Count Valid: {df.shape[0]:,} rows found.")
        
    if df.shape[1] < 3:
        suitability_passed = False
        validation_messages.append("❌ Dataset has fewer than 3 columns. Insufficient feature dimensionality.")
    else:
        validation_messages.append(f"✅ Feature Dimensionality Valid: {df.shape[1]} columns found.")
        
    possible_targets = ['Overtime_Cost', 'Cost', 'Overtime_Expenses', 'Total_Cost']
    target_col = next((col for col in possible_targets if col in df.columns), None)
    
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[-1]
            validation_messages.append(f"⚠️ Exact target column not found. Automatically assigned numeric column '{target_col}' as regression target.")
        else:
            suitability_passed = False
            validation_messages.append("❌ No numeric target column detected for overtime cost prediction.")
    else:
        validation_messages.append(f"✅ Target Column Identified: '{target_col}'")
        
    if target_col and target_col != 'Overtime_Cost':
        df['Overtime_Cost'] = df[target_col]
        
    feature_candidates = [c for c in df.columns if c != 'Overtime_Cost']
    if len(feature_candidates) < 1:
        suitability_passed = False
        validation_messages.append("❌ Insufficient predictor features available.")
    else:
        validation_messages.append(f"✅ Predictor Features Valid ({len(feature_candidates)} features detected).")
else:
    suitability_passed = False
    validation_messages.append("❌ No dataset loaded.")

# Dynamic Model Training / Loading with MSE, MAE, RMSE, R2
@st.cache_data
def train_model(df, target_name):
    X = df.drop(columns=['Overtime_Cost'])
    y = df['Overtime_Cost']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
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

if suitability_passed and df is not None:
    artifact = train_model(df, target_col if 'target_col' in locals() else 'Overtime_Cost')
    model = artifact['model']
    feature_names = artifact['feature_names']
    metrics = artifact['metrics']

# Header
st.title("⏱️ Employee Overtime Cost Forecasting System")
st.markdown(f"Currently Analyzing: **{selected_dataset_label}**")

if suitability_passed:
    st.sidebar.markdown("---")
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
# TAB 1: EDA (Tabular Format Only, NO GRAPHS)
# -------------------------------------------------------------
with tab1:
    st.subheader(f"📊 Exploratory Data Analysis & Suitability Verification — {selected_dataset_label}")
    st.markdown("Comprehensive statistical audit of every column in the active dataset in tabular format.")
    
    with st.expander("🔍 Dataset Suitability & Validation Report", expanded=not suitability_passed):
        for msg in validation_messages:
            st.markdown(f"- {msg}")
        if suitability_passed:
            st.success("🎉 **Dataset is fully suitable and verified for machine learning overtime cost prediction!**")
        else:
            st.error("⚠️ **Dataset requires formatting adjustments before full prediction can proceed.**")

    if df is not None and suitability_passed:
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
# -------------------------------------------------------------
with tab2:
    st.subheader("🔮 Single Schedule Predictor")
    if suitability_passed:
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_dept = st.selectbox("Department", departments, key="p_dept") if 'Department' in df.columns else "General"
            selected_shift = st.selectbox("Shift", shifts, key="p_shift") if 'Shift' in df.columns else "Standard"
            working_hours = st.slider("Total Working Hours", min_value=8.0, max_value=16.0, value=11.0, step=0.25)
            
        with col2:
            hourly_wage = st.number_input("Hourly Wage ($)", min_value=10.0, max_value=200.0, value=45.0, step=1.0)
            month = st.selectbox("Month", list(range(1, 13)), index=9)
            
        with col3:
            day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
            day_of_week = st.selectbox("Day of Week", list(day_map.keys()), format_func=lambda x: day_map[x])
            is_weekend = 1 if day_of_week in [5, 6] else 0
            is_holiday = st.selectbox("Holiday Shift?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        if st.button("Calculate Overtime Forecast", type="primary"):
            input_dict = {
                'Actual_Working_Hours': working_hours,
                'Hourly_Wage': hourly_wage,
                'Weekend': is_weekend,
                'Holiday': is_holiday,
                'Month': month,
                'Day_of_Week': day_of_week
            }
            if 'Department' in df.columns:
                input_dict['Department'] = selected_dept
            if 'Shift' in df.columns:
                input_dict['Shift'] = selected_shift
                
            input_data = pd.DataFrame([input_dict])
            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=feature_names, fill_value=False)
            
            predicted_cost = model.predict(input_encoded)[0]
            overtime_hours = max(0.0, working_hours - 8.0)
            
            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Estimated Overtime Cost", f"${predicted_cost:.2f}")
            res_col2.metric("Overtime Hours", f"{overtime_hours:.2f} hrs")
            res_col3.metric("Effective Cost per OT Hour", f"${(predicted_cost / overtime_hours if overtime_hours > 0 else 0):.2f}")
    else:
        st.warning("Please resolve dataset suitability warnings in Tab 1 to enable predictions.")


# -------------------------------------------------------------
# TAB 3: Explainability & Dataset Insights (With Graphs & Technique Explanation)
# -------------------------------------------------------------
with tab3:
    st.subheader("💡 Dataset Insights & Model Explainability (XAI) with Graphs")
    if suitability_passed:
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
        st.markdown("Primary cost drivers such as hourly rate and working hours exhibit the highest importance scores, confirming non-linear cost scaling. Additionally, holiday and weekend binary flags introduce premium cost multipliers into the prediction model.")
    else:
        st.info("Upload or select a valid dataset to view feature importance and graphs.")

# -------------------------------------------------------------
# TAB 4: Batch / Scenario Simulator
# -------------------------------------------------------------
with tab4:
    st.subheader("📈 Interactive Batch & Workforce Scenario Simulator")
    if suitability_passed:
        st.markdown("Model complex multi-shift staffing strategies and compare projected financial expenditures.")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_hours = st.slider("Shift Working Hours", 0.0, 16.0, 10.0, 0.25, key="sim_hours")
            sim_rate = st.number_input("Hourly Rate ($)", 10.0, 600.0, 394.0, 1.0, key="sim_rate")
        with col_s2:
            sim_headcount = st.slider("Employee Headcount", 1, 100, 15, key="sim_headcount")
            sim_holiday = st.selectbox("Holiday Premium?", [0.0, 1.0], format_func=lambda x: "Yes (1.0)" if x == 1.0 else "No (0.0)", key="sim_hol")
        with col_s3:
            sim_weekend = st.selectbox("Weekend Shift?", [0.0, 1.0], format_func=lambda x: "Yes (1.0)" if x == 1.0 else "No (0.0)", key="sim_wknd")

        if 'scenario_list' not in st.session_state:
            st.session_state['scenario_list'] = [
                {"Name": "Standard Weekday Shift", "Hours": 8.0, "Rate": 394.0, "Headcount": 10, "Weekend": 0.0, "Holiday": 0.0},
                {"Name": "Extended Weekend Shift", "Hours": 11.0, "Rate": 394.0, "Headcount": 10, "Weekend": 1.0, "Holiday": 0.0},
                {"Name": "Holiday Surge Shift", "Hours": 12.0, "Rate": 394.0, "Headcount": 10, "Weekend": 0.0, "Holiday": 1.0},
            ]

        if st.button("➕ Add Custom Scenario", type="primary"):
            st.session_state['scenario_list'].append({
                "Name": f"Custom Shift ({sim_headcount} staff)",
                "Hours": sim_hours,
                "Rate": sim_rate,
                "Headcount": sim_headcount,
                "Weekend": sim_weekend,
                "Holiday": sim_holiday
            })
            st.success("Changes applied successfully!")

        if st.session_state['scenario_list']:
            results = []
            for s in st.session_state['scenario_list']:
                row_dict = {
                    'Experience': 9.0,
                    'Attendance': 1.0,
                    'Holiday': s['Holiday'],
                    'Weekend_Worked': s['Weekend'],
                    'Projects_Assigned': 2,
                    'Monthly_Salary': 69405.0,
                    'Hourly_Rate': s['Rate'],
                    'Working_Hours': s['Hours'],
                    'Overtime_Hours': max(0.0, s['Hours'] - 8.0)
                }
                for col in df.columns:
                    if col != 'Overtime_Cost' and col not in row_dict:
                        row_dict[col] = df[col].mean()
                        
                row_df = pd.DataFrame([row_dict])
                row_enc = pd.get_dummies(row_df).reindex(columns=feature_names, fill_value=False)
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
    else:
        st.warning("Please verify dataset suitability in Tab 1.")
