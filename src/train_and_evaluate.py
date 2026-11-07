"""
Train and evaluate Multiple Linear Regression, Decision Tree, and Random Forest models
for Air Quality Index (AQI) prediction from pollutant levels.
Generates metrics, feature importances, and analytical plots.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "aqi_pollutant_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_aqi.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

FEATURE_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET_COL = "AQI"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    # Stratified or consistent random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Models to benchmark
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results[name] = {
            "R2_Score": round(float(r2), 4),
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4)
        }
        fitted_models[name] = model
        print(f"\n--- {name} ---")
        print(f"R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # Focus on best model: Random Forest
    best_model = fitted_models["Random Forest Regressor"]
    y_test_pred = best_model.predict(X_test)
    
    # Save the trained model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Calculate feature importances
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    results["Feature_Importances"] = dict(zip(feat_df["Feature"], feat_df["Importance"].round(4)))
    
    # Find sample predictions: 1 highly accurate, 1 with highest error
    test_eval_df = X_test.copy()
    test_eval_df["Actual_AQI"] = y_test.values
    test_eval_df["Predicted_AQI"] = np.round(y_test_pred, 2)
    test_eval_df["Absolute_Error"] = np.round(np.abs(test_eval_df["Actual_AQI"] - test_eval_df["Predicted_AQI"]), 2)
    test_eval_df["Percentage_Error"] = np.round((test_eval_df["Absolute_Error"] / test_eval_df["Actual_AQI"]) * 100, 2)
    
    best_sample = test_eval_df.sort_values("Absolute_Error").iloc[0].to_dict()
    worst_sample = test_eval_df.sort_values("Absolute_Error", ascending=False).iloc[0].to_dict()
    
    results["Sample_Accurate_Case"] = best_sample
    results["Sample_Divergent_Case"] = worst_sample
    
    # Save JSON results
    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {REPORTS_DIR / 'model_metrics.json'}")
    
    # ------------------ PLOTS ------------------
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.4, color='#1f77b4', edgecolors='none', label='Test Instances')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal 1:1 Identity')
    plt.title("Actual vs. Predicted AQI (Random Forest)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Actual AQI (Ground Truth)", fontsize=11)
    plt.ylabel("Predicted AQI (Random Forest)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close()
    
    # Plot 2: Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Pollutant Feature Importance (Mean Decrease in Impurity)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Pollutant", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=200)
    plt.close()
    
    # Plot 3: Residual Distribution
    residuals = y_test - y_test_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, bins=30, color='#2ca02c')
    plt.axvline(0, color='red', linestyle='--', lw=1.5)
    plt.title("Residual Distribution (Actual - Predicted AQI)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Prediction Error (AQI Points)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_distribution.png", dpi=200)
    plt.close()
    
    print("All figures successfully saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
