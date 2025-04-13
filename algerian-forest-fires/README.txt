Algerian Forest Fires: Forecasting Fire Risk Using Regression

This project uses regression techniques to predict the **Fire Weather Index (FWI)** — a numeric indicator of fire danger — based on environmental data collected from two Algerian regions between June and September 2012.

Objective
Develop a predictive model for FWI to help anticipate high-risk fire days and support emergency planning.

Dataset Summary
1. 244 instances (122 from Bejaia, 122 from Sidi Bel-Abbes)
2. Features: Temp, RH, Wind, Rain, FFMC, DMC, DC, ISI, BUI, Region, Classes
3. Target: Fire Weather Index (FWI)

Workflow Overview
1. Data cleaning and preprocessing
2. Exploratory Data Analysis
3. Feature scaling with StandardScaler
4. Model training using Linear, Lasso, and Ridge Regression
5. Exporting best model (Ridge) for deployment

Results

| Model            | R² Score | MAE   |
|------------------|----------|-------|
| Linear Regression| 0.989    | 0.465 |
| Ridge Regression | 0.987    | 0.503 |
| Lasso Regression | 0.954    | 1.08  |

How to Use

import pickle
import numpy as np

# Load model and scaler
ridge_model = pickle.load(open("ridge_model.pickle", "rb"))
scaler = pickle.load(open("scaler.pickle", "rb"))

# Sample input
X_sample = np.array([[32, 38, 18, 0.0, 88.0, 40.0, 120.0, 12.0, 52.3, 1, 1]])
X_scaled = scaler.transform(X_sample)

# Predict
print(ridge_model.predict(X_scaled))
