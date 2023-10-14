# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:13:27 2023

@author: DELL
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Specify the 'latin1' encoding parameter
data = pd.read_csv('C:/oasis intern/spam.csv', encoding='latin1')

# Check the structure of your data
print(data.head())

# Assuming 'v2' is the column containing email text
X = data['v2']
y = data['v1']

# Data splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text preprocessing and feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Label encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Preprocessing the text data
def preprocess_text(text):
    # Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Tokenization and lowercasing
    tokens = text.lower().split()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming (reducing words to their root form)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join the tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

X_train_preprocessed = [preprocess_text(text) for text in X_train]
X_test_preprocessed = [preprocess_text(text) for text in X_test]

# Optional: Feature selection using chi-squared test (you can adjust k)
k_best = SelectKBest(chi2, k=1000)
X_train_selected = k_best.fit_transform(X_train_tfidf, y_train_encoded)
X_test_selected = k_best.transform(X_test_tfidf)


from sklearn.preprocessing import LabelEncoder

# Create a label encoder
label_encoder = LabelEncoder()

# Fit the label encoder on your labels (v1 column in your DataFrame)
label_encoder.fit(data['v1'])

# Encode the labels in your DataFrame
data['v1_encoded'] = label_encoder.transform(data['v1'])

# Display the updated DataFrame with encoded labels
print(data.head())

from sklearn.model_selection import train_test_split

# Define your features (X) and target (y)
X = data['v2']  # Assuming 'v2' is your feature column (email_text)
y = data['v1_encoded']  # 'v1_encoded' is the encoded label column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the vectorizer on your training email text data
X_train_features = tfidf_vectorizer.fit_transform(X_train)

# Now, X_train_features contains the TF-IDF features for your training data

from sklearn.naive_bayes import MultinomialNB

# Create an instance of the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model using your TF-IDF features (X_train_features) and labels (y_train)
model.fit(X_train_features, y_train)

# Assuming you already have a TF-IDF vectorizer (tfidf_vectorizer) fitted on your training data
X_test_features = tfidf_vectorizer.transform(X_test)

y_pred = model.predict(X_test_features)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(X_test_features)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", confusion)





