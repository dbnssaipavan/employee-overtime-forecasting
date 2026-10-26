import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import matplotlib.pyplot as plt

def train_forecasting_model(data_path="overtime_dataset_single_columns.csv"):
    print("Loading historical overtime dataset...")
    df = pd.read_csv(data_path)

    # Separate Features and Target
    X = df[['Department', 'Shift', 'Actual_Working_Hours', 'Hourly_Wage', 
            'Weekend', 'Holiday', 'Month', 'Day_of_Week']].copy()
    y = df['Overtime_Cost']

    # One-Hot Encode Categorical Features
    X_encoded = pd.get_dummies(X, columns=['Department', 'Shift'], drop_first=True)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Train Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R2: {r2:.4f}")

    # Generate True SHAP Global Summary Plot
    print("Generating SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Impact on Overtime Cost", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    plt.close()
    print("SHAP summary plot saved as 'shap_summary.png'.")

    # Save Artifacts
    model_artifact = {
        'model': model,
        'feature_names': X_encoded.columns.tolist(),
        'departments': sorted(df['Department'].unique().tolist()),
        'shifts': sorted(df['Shift'].unique().tolist()),
        'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    }
    joblib.dump(model_artifact, "rf_model.pkl")
    print("Model saved as 'rf_model.pkl'.")

if __name__ == "__main__":
    train_forecasting_model()
