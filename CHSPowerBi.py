import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

# Prepare the data
X_categorical = pd.get_dummies(dataset['Gender'])  # One-hot encode 'Gender'
X_numerical = dataset[['Age', 'EstimatedSalary']]  # Other numerical features
X = pd.concat([X_categorical, X_numerical], axis=1)  # Concatenate categorical and numerical features
y = dataset['Purchased']  # Target variable

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Predict outcomes
predicted_outcomes = model.predict(X)

# Add predicted outcomes as a new column
dataset['Predicted_Outcome'] = predicted_outcomes