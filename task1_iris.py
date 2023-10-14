# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:31:48 2023

@author: DELL
"""

import pandas as pd

# Load the dataset
df = pd.read_csv('C:\oasis intern\Iris.csv')
# Display the first few rows of the dataset
print(df.head())
# Summary statistics for numerical features
print(df.describe())
# Check data types
print(df.dtypes)
# Check for missing values
print(df.isnull().sum())
import matplotlib.pyplot as plt

# Create box plots for numerical features
df.boxplot()
plt.show()
df.hist()
plt.show()
import seaborn as sns
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

sns.pairplot(df, hue='Species')
plt.show()


# Use 'Species' with an uppercase 'S'
sns.boxplot(x='Species', y='SepalLengthCm', data=df)
plt.show()
sns.countplot(x='Species', data=df)
plt.show()

from sklearn.model_selection import train_test_split

# Define your features (X) and target variable (y)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the dataset into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the dimensions of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance on the testing data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# You can also print more detailed metrics like classification report and confusion matrix
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
