import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import matplotlib.pyplot as plt

def train_forecasting_model(data_path="overtime_forecast_dataset.csv"):
    # 1. Load Dataset
    print("Loading historical overtime dataset...")
    df = pd.read_csv(data_path)

    # 2. Separate Features and Target
    X = df[['Department', 'Shift', 'Actual_Working_Hours', 'Hourly_Wage', 
            'Weekend', 'Holiday', 'Month', 'Day_of_Week']].copy()
    y = df['Overtime_Cost']

    # 3. One-Hot Encode Categorical Features for Model Training
    X_encoded = pd.get_dummies(X, columns=['Department', 'Shift'], drop_first=True)

    # 4. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 5. Train Regression Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE) : ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared Score (R2)      : {r2:.4f}")

    # 7. Explainable AI (SHAP Analysis)
    print("\nGenerating SHAP Summary Plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    print("Saved SHAP summary plot as 'shap_summary.png'.")

    # 8. Save Trained Model and Metadata Artifacts
    model_artifact = {
        'model': model,
        'feature_names': X_encoded.columns.tolist(),
        'departments': sorted(df['Department'].unique().tolist()),
        'shifts': sorted(df['Shift'].unique().tolist()),
        'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    }
    
    joblib.dump(model_artifact, "rf_model.pkl")
    print("Model saved successfully as 'rf_model.pkl'.")

if __name__ == "__main__":
    train_forecasting_model()