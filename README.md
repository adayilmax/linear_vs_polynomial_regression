# Linear and Polynomial Regression: Comparison on Synthetic Datasets

## Overview
This project implements linear regression (scikit-learn, closed-form OLS, and
gradient descent) on a linear dataset, and polynomial regression on a nonlinear
dataset, comparing performance with mean squared error (MSE).

## Data
- Dataset 1: synthetic linear data with Gaussian noise
- Dataset 2: synthetic nonlinear data (provided as `.npy`)
- Split: 50% train / 50% validation

## Methods
- Linear regression:
  - scikit-learn `LinearRegression`
  - manual OLS (pseudoinverse)
  - manual gradient descent
- Polynomial regression:
  - scikit-learn `PolynomialFeatures` + `LinearRegression`, degrees {1, 3, 5, 7}
  - manual polynomial (degree 3) via pseudoinverse

## Results (validation MSE)
- Linear regression:
  - scikit-learn ≈ 0.00795
  - OLS (pinv) ≈ 0.00795
  - gradient descent ≈ 0.0026 (training loss trajectory shown; converges near closed-form)
- Polynomial regression (Dataset 2):
  - deg=1: ≈ 0.06363
  - deg=3: ≈ 0.01206
  - deg=5: ≈ 0.00748
  - deg=7: ≈ 0.01155

## Reproducibility
```bash
pip install -r requirements.txt
jupyter notebook notebooks/linear_vs_polynomial_regression.ipynb
