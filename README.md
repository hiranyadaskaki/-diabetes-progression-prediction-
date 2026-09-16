# -diabetes-progression-prediction-
Predicting 1-year diabetes disease progression using scikit-learn Linear Regression.
# Diabetes Progression Prediction 

## Objective Predict quantitative disease progression one year after baseline using patient health measurements from the `scikit-learn` diabetes dataset. 

## Completed Tasks 
1. **Dataset Loading**: Loaded the `load_diabetes()` dataset containing 442 patient records and 10 baseline clinical features (age, sex, BMI, blood pressure, and 6 blood serum measurements).
2. **Exploratory Data Analysis (EDA)**: Verified dataset shape, confirmed zero missing values, and calculated summary statistics.
3. **Feature Correlation Analysis**: Analyzed feature correlations against disease progression, finding BMI (0.586) and s5 serum measurement (0.566) as the top predictors.
4. **Model Training**: Trained a `LinearRegression` model using an 80/20 train-test split.
5. **Model Evaluation**: Assessed model accuracy using standard metrics: - **R² Score**: `0.4526` (Explains ~45.3% of target variance) - **Mean Absolute Error (MAE)**: `42.79` - **Root Mean Squared Error (RMSE)**: `53.85`
6. **Influential Features**: Identified key feature weights (`s5` serum level, `bmi`, `s2` LDL, and `bp` blood pressure).
7. **Actual vs. Predicted Evaluation**: Evaluated residuals and compared predicted values against actual patient targets.

## Visualizations ![Dashboard Visualizations](diabetes\_project\_visualizations.png) ## How to Run \`\`\`bash pip install -r requirements.txt python diabetes\_progression\_project.py
