# Air Quality Index (AQI) Prediction from Ambient Pollutant Concentrations

An end-to-end Machine Learning Case Study predicting continuous Air Quality Index (AQI) values and assessing severity categories from multi-pollutant sensor measurements ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$).

## Project Highlights
- **Problem Formulation**: Supervised Machine Learning — Regression & Environmental Impact Assessment.
- **Algorithms Benchmarked**: Multiple Linear Regression (Baseline), Decision Tree Regressor, and Random Forest Ensemble.
- **Top Model Performance**: Random Forest achieves an $R^2$ of **0.9958** with an MAE of **2.39** AQI points on holdout test data.
- **Interactive UI**: Real-time Streamlit web interface for pollutant entry and health advisory mapping.

## Repository Structure
```
air_quality_prediction/
├── data/
│   └── aqi_pollutant_dataset.csv     # 5,000 multi-pollutant monitoring records
├── models/
│   └── random_forest_aqi.joblib      # Serialized scikit-learn model
├── reports/
│   ├── figures/                      # Evaluation plots (Actual vs Pred, Feature Importance)
│   ├── model_metrics.json            # Numerical evaluation metrics
│   └── CLA3_Case_Study_Report.md     # Full student case study report
├── src/
│   ├── generate_dataset.py           # Dataset synthesis based on CPCB/EPA standards
│   └── train_and_evaluate.py         # Model training, validation, and benchmarking
├── app.py                            # Streamlit web application
└── README.md
```

## Quick Start
```bash
# 1. Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn streamlit joblib

# 2. Run training & evaluation
python air_quality_prediction/src/train_and_evaluate.py

# 3. Launch Streamlit App
streamlit run air_quality_prediction/app.py
```
