# california-housing-ml-regression
Machine learning regression project using the California Housing dataset. Implements Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and SVR models with preprocessing, feature scaling, evaluation using MSE, MAE, and R², and performance comparison.

🏠 California Housing Regression Analysis

📌 Project Overview

This project demonstrates the application of regression techniques in supervised machine learning using the California Housing dataset from scikit-learn. Multiple regression models are implemented, evaluated, and compared to identify the best-performing model for predicting median house prices.

🎯 Objective

The objective of this project is to:

- Understand and apply different regression algorithms

- Perform data preprocessing and feature scaling

- Evaluate models using standard regression metrics

- Compare model performance and justify results

📊 Dataset

- Source: fetch_california_housing() from sklearn.datasets

Description:
- The dataset contains information about housing features in California such as median income, house age, average rooms, population, and location-based attributes.

- Target Variable: MedHouseValue (Median house value)

⚙️ Technologies Used

* Python

* Pandas

* Scikit-learn

🔄 Machine Learning Pipeline

- Data Loading

- Data Preprocessing

- Conversion to Pandas DataFrame

- Missing value check

- Feature scaling using StandardScaler

- Train–Test Split

- Model Training

- Model Evaluation

- Model Comparison

🤖 Regression Models Implemented

- Linear Regression

- Decision Tree Regressor

- Random Forest Regressor

- Gradient Boosting Regressor

- Support Vector Regressor (SVR)

Each model is trained and evaluated on the same dataset for fair comparison.

📈 Model Evaluation Metrics

The models are evaluated using:

* Mean Squared Error (MSE)

* Mean Absolute Error (MAE)

* R-squared Score (R²)

- Lower MSE and MAE values and higher R² values indicate better model performance.

🏆 Results Summary

Best Performing Model: Random Forest Regressor

- Lowest error values

- Highest R² score (~80%)

Worst Performing Model: Linear Regression

- Higher error values

- Limited ability to capture non-linear relationships

Ensemble models performed better due to their ability to handle complex patterns in the data.
