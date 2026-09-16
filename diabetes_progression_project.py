"""
===============================================================================
DIABETES PROGRESSION PREDICTION PROJECT
Dataset: scikit-learn load_diabetes()
Objective: Predict quantitative disease progression score after 1 year based
           on baseline patient clinical health measurements.

Tasks completed in this script:
1. Load dataset (scikit-learn)
2. Exploratory Data Analysis (EDA)
3. Feature correlation analysis
4. Train Linear Regression model
5. Model evaluation (R², MAE, RMSE)
6. Identify most influential features
7. Actual vs Predicted comparison & Residual Analysis
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    feature_names = list(diabetes.feature_names)
    target_name = 'target'

    print(f"Dataset Shape: {df.shape} (442 patients, 10 features + 1 target)")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n" + "=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print("Missing values per column:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())

    print("\n" + "=" * 60)
    print("3. FEATURE CORRELATIONS WITH TARGET")
    print("=" * 60)
    corr = df.corr()
    print(corr[target_name].sort_values(ascending=False))

    print("\n" + "=" * 60)
    print("4. MODEL TRAINING (LINEAR REGRESSION)")
    print("=" * 60)
    X = df[feature_names]
    y = df[target_name]

    # Split dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model Intercept: {model.intercept_:.2f}")

    print("\n" + "=" * 60)
    print("5. MODEL EVALUATION")
    print("=" * 60)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score (Coefficient of Determination) : {r2:.4f}")
    print(f"MAE (Mean Absolute Error)               : {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error)          : {rmse:.4f}")

    print("\n" + "=" * 60)
    print("6. INFLUENTIAL FEATURES ANALYSIS")
    print("=" * 60)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print("Model Coefficients (Sorted by Absolute Impact):")
    print(coef_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("7. ACTUAL VS PREDICTED SAMPLE COMPARISON (First 10 Test Patients)")
    print("=" * 60)
    comparison_df = pd.DataFrame({
        'Actual Progression': y_test.values[:10],
        'Predicted Progression': np.round(y_pred[:10], 1),
        'Difference (Residual)': np.round(y_test.values[:10] - y_pred[:10], 1)
    })
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
