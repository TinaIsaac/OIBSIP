# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:20:30 2023

@author: DELL
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your dataset
df = pd.read_csv('C:/oasis intern/car data.csv')  # Replace with your dataset file path

# Display the first few rows of the dataset
print(df.head())

# Get the column titles as a list
column_titles = df.columns.tolist()
print(column_titles)

# Data Cleaning
# 1. Handling Missing Values
# Check for missing values in the dataset
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# In this example, let's remove rows with missing values

# 2. Handling Outliers
# Visualize outliers using box plots for numeric columns
numeric_columns = ['Year', 'Selling_Price', 'Present_Price', 'Driven_kms']
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=col, data=df)

# Explore relationships between features and the target variable (Selling_Price).

# Scatter plot of Year vs. Selling_Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Selling_Price', data=df)

# Box plot of Fuel_Type vs. Selling_Price
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df)

# 3. Feature Engineering (if needed)
df_encoded = pd.get_dummies(df, columns=['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

# 'drop_first=True' removes one of the dummy variables to avoid multicollinearity.

# Display the first few rows of the encoded dataset
print(df_encoded.head())

# Separate the target variable (Selling_Price) from the features
X = df_encoded.drop('Selling_Price', axis=1)  # Features
y = df_encoded['Selling_Price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


