# Employee Overtime Cost Forecasting System

An end-to-end Machine Learning and Explainable AI (XAI) web application designed to forecast employee overtime expenses, optimize workforce scheduling, and empower HR and operations managers with data-driven budgeting insights.

---

## 🚀 Key Features

* **Predictive Modeling:** Built using a tuned `RandomForestRegressor` to accurately predict overtime costs based on workforce schedule parameters.
* **Explainable AI (SHAP):** Integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, game-theory-backed global and local feature attributions, explaining *why* a specific shift incurs high expenses.
* **Interactive Streamlit Dashboard:** A multi-tab web interface featuring:
  * **Single Schedule Predictor:** Instant cost calculations and local SHAP factor breakdowns.
  * **Batch Scenario Simulator:** Side-by-side comparison of team costs under different scheduling options (e.g., standard vs. extended weekend shifts).
  * **Global XAI Dashboard:** Visual summaries of the top drivers influencing labor costs.
* **Automatic Cloud Fallback:** Seamlessly handles model loading and auto-trains on the fly if pre-trained artifacts are missing during deployment.

---

## 📁 Project Structure

```text
employee-overtime-forecasting/
│
├── app.py                         # Interactive Streamlit web application
├── train_model.py                 # Model training, evaluation (MAE/RMSE), and SHAP generation script
├── overtime_dataset_single_columns.csv # Historical workforce dataset
├── rf_model.pkl                   # Saved model artifact and metadata
├── shap_summary.png               # Global SHAP feature impact visualization
└── requirements.txt               # Python package dependencies
