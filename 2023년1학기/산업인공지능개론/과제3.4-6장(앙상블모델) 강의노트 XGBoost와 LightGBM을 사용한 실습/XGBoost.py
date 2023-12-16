import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load the Boston Housing dataset (version 1) and set parser='auto'
boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
data = boston.frame
data['PRICE'] = boston.target

# Display the first few rows of the dataset
print(data.head())

# Separate the features (X) and the target variable (y)
X, y = data.drop(columns=['PRICE']), data['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Identify the categorical columns
categorical_cols = ['CHAS', 'RAD']

# Convert the categorical columns to 'category' dtype
X_train[categorical_cols] = X_train[categorical_cols].astype('category')
X_test[categorical_cols] = X_test[categorical_cols].astype('category')

# Create an XGBoost DMatrix with the categorical feature enabled
train_dmatrix = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
test_dmatrix = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)

# Set the XGBoost hyperparameters
params = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}

# Train the XGBoost model
xg_reg = xgb.train(params=params, dtrain=train_dmatrix)

# Make predictions on the test data
preds = xg_reg.predict(test_dmatrix)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))