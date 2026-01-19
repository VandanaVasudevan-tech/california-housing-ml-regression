# ======================================================================================================================
# Load Required Libraries
# ======================================================================================================================
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import pandas as pd

# ======================================================================================================================
# STEP 1: Load the Dataset
# The California Housing dataset is loaded using sklearn.
# This dataset contains numerical features related to
# housing characteristics and a continuous target variable.
# ======================================================================================================================
print("Step 1: Loading California Housing Dataset")
data = fetch_california_housing()
print("Dataset loaded successfully")

# ======================================================================================================================
# STEP 2: DATAFRAME CREATION
# Features are converted into a pandas DataFrame to
# improve readability and allow easy data manipulation.
# The target variable is kept separate to avoid data leakage.
# ======================================================================================================================
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="MedHouseValue")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ======================================================================================================================
# STEP 3: Combine Features and Target (For Visualization)
# ======================================================================================================================
print("\nStep 3: Creating combined DataFrame for visualization")
df = pd.concat([X, y], axis=1)
print(df.head())

# ======================================================================================================================
# STEP 4: Check for Missing Values
# Checking for missing values is a mandatory preprocessing
# step to ensure data quality. Missing values can negatively
# impact model performance and cause errors during training.
# ======================================================================================================================
print("\nStep 4: Checking for missing values")
print(X.isnull().sum())

# ======================================================================================================================
# STEP 5: Feature Scaling
# The dataset contains features with different magnitudes
# Standardization is applied
# to bring all features to a common scale with mean 0 and
# standard deviation 1.
# This is necessary to ensure fair contribution of all
# features and stable regression performance.
# ======================================================================================================================
print("\nStep 5: Applying Standardization")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaled_data, columns=X.columns)
print("Feature scaling completed")
print(X_scaled.head())

print("Interpretation: \n The preprocessing steps included converting the dataset into a pandas DataFrame, checking "
      "for missing values, and applying feature scaling using standardization. Although no missing values were found, "
      "this step ensures data quality and robustness. Feature scaling was necessary because the dataset contains "
      "features with varying ranges, and standardization ensures that all features contribute equally to the regression"
      " model and improves training stability.")
# ======================================================================================================================

# LINEAR REGRESSION IMPLEMENTATION

# ======================================================================================================================
# STEP 1: TRAIN-TEST SPLIT
# The dataset is split into training and testing sets
# to evaluate the model's performance on unseen data.
# ======================================================================================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Train-test split completed")

# ======================================================================================================================
# STEP 2: MODEL CREATION AND TRAINING
# Linear Regression models the relationship between input
# features and the target variable by fitting a linear
# equation that minimizes the sum of squared errors between
# predicted and actual values.

# WHY SUITABLE :
# It serves as a strong baseline model and helps understand linear
# relationships between features like median income and house price.
# ======================================================================================================================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Linear Regression model trained successfully")

# ======================================================================================================================
# STEP 3: MAKE PREDICTION ON TEST DATA
# ======================================================================================================================
y_pred_lr = lr_model.predict(X_test)

# ======================================================================================================================
# STEP 4: MODEL EVALUATION
# Mean Squared Error , Mean Absolute Error and R² score are
# used to measure prediction accuracy and model goodness
# of fit.
# ======================================================================================================================
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression MAE: {mae_lr}")
print(f"Linear Regression R2 Score: {r2_lr}")


# DECISION TREE REGRESSOR IMPLEMENTATION


# ======================================================================================================================
# STEP 2: MODEL CREATION AND TRAINING
# Decision Tree learns non-linear patterns by recursively
# splitting the data to minimize prediction error.
# ======================================================================================================================
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree Regressor trained successfully")

# ======================================================================================================================
# STEP 3: MODEL PREDICTION AND EVALUATION
# Mean Squared Error and R² score are used to measure
# prediction accuracy and model goodness of fit.
# ======================================================================================================================
y_pred = dt_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Decision Tree Performance:")
print(f"Decision Tree MSE: {mse}")
print(f"Decision Tree R² Score: {r2}")
print(f"Decision Tree MAE: {mae}")


# RANDOM FOREST REGRESSOR IMPLEMENTATION


# ======================================================================================================================
# STEP 2: MODEL CREATION AND TRAINING
# Random Forest Regressor is an ensemble learning algorithm
# that builds multiple decision trees using different subsets
# of the data and features. The final prediction is obtained by
# averaging the predictions of all individual trees, which reduces
# overfitting and improves accuracy.

# WHY SUITABLE :
# The California Housing dataset contains complex,
# non-linear relationships between features such as income, population,
# and location. Random Forest is well suited because it captures non-linear
# patterns, handles feature interactions effectively, and provides more
# stable predictions than a single decision tree.
# ======================================================================================================================
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully")

# ======================================================================================================================
# STEP 3: MAKE PREDICTION ON TEST DATA
# ======================================================================================================================
rf_pred = rf_model.predict(X_test)

# ======================================================================================================================
# STEP 4: MODEL EVALUATION
# Mean Squared Error , Mean Absolute Error and R² score are
# used to measure prediction accuracy and model goodness
# of fit.
# ======================================================================================================================
mse = mean_squared_error(y_test, rf_pred)
r2 = r2_score(y_test, rf_pred)
mae = mean_absolute_error(y_test, rf_pred)
print("Random Forest Performance:")
print(f"Random Forest MSE: {mse}")
print(f"Random Forest R² Score: {r2}")
print(f"Random Forest MAE: {mae}")


# SUPPORT VECTOR REGRESSOR IMPLEMENTATION


# ======================================================================================================================
# STEP 2: MODEL CREATION AND TRAINING
# Support Vector Regressor attempts to find a function that
# deviates from the actual target values by no more than a specified
# margin while keeping the model as simple as possible. It can model
# non-linear relationships using kernel functions.

# WHY SUITABLE :
# SVR is suitable for medium-sized datasets and can handle non-linear
# patterns present in housing price data. When combined with feature
# scaling, it can produce accurate predictions by controlling model
# complexity.
# ======================================================================================================================
svr_model = SVR()
svr_model.fit(X_train, y_train)
print("SVR model trained successfully")
# ======================================================================================================================
# STEP 3: MAKE PREDICTION ON TEST DATA
# ======================================================================================================================
svr_pred = svr_model.predict(X_test)
# ======================================================================================================================
# STEP 4: MODEL EVALUATION
# Mean Squared Error , Mean Absolute Error and R² score are
# used to measure prediction accuracy and model goodness
# of fit.
# ======================================================================================================================
svr_mse = mean_squared_error(y_test, svr_pred)
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)
print("SVR Performance:")
print(f"SVR MSE: {svr_mse}")
print(f"SVR R² Score: {svr_r2}")
print(f"SVR MAE: {svr_mae}")
# ======================================================================================================================


# GRADIENT BOOSTING REGRESSOR IMPLEMENTATION


# ======================================================================================================================
# STEP 2: MODEL CREATION AND TRAINING
# Gradient Boosting is an ensemble learning method that builds models sequentially. Each new decision tree is trained
# to correct the errors (residuals) made by the previous trees. The final prediction is the sum of predictions from all
# trees.

# WHY SUITABLE :
# Captures complex non-linear relationships
# Handles feature interactions effectively
# Performs well on structured/tabular data
# Reduces bias compared to single models like Linear Regression
# Hence, Gradient Boosting is well-suited for predicting housing prices, which depend on multiple interacting factors.
# ======================================================================================================================
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
print("Gradient Boosting model trained successfully")
# ======================================================================================================================
# STEP 3: MAKE PREDICTION ON TEST DATA
# ======================================================================================================================
gbr_pred = gbr.predict(X_test)
# ======================================================================================================================
# STEP 4: MODEL EVALUATION
# Mean Squared Error , Mean Absolute Error and R² score are
# used to measure prediction accuracy and model goodness
# of fit.
# ======================================================================================================================
gbr_mse = mean_squared_error(y_test, gbr_pred)
gbr_mae = mean_absolute_error(y_test, gbr_pred)
gbr_r2 = r2_score(y_test, gbr_pred)
print("Gradient Boosting Performance:")
print(f"Gradient Boosting MSE: {gbr_mse}")
print(f"Gradient Boosting R² Score: {gbr_r2}")
print(f"Gradient Boosting MAE: {gbr_mae}")
# ======================================================================================================================

# PREDICTION
print("""PREDICTION:\nBased on the evaluation metrics, the Random Forest Regressor is the best-performing model for the 
California Housing dataset. It achieved the lowest MSE (0.255), lowest MAE (0.328), and the highest R² score (0.805), 
indicating strong predictive accuracy and good generalization.
The Linear Regression model performed the worst, with the highest error values and the lowest R² score (0.575).This is
because linear regression assumes a linear relationship,whereas the housing dataset contains complex non-linear patterns.
Therefore, ensemble models like Random Forest are more suitable for this dataset.""")
