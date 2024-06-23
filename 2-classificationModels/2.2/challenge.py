# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the heart dataset
heart = pd.read_csv('heart.csv')

# Create a dataframe X containing thalach and age
X = heart[['thalach', 'age']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Output feature: target
y = heart[['target']]

# Initialize the model
heartLogReg = LogisticRegression(penalty=None)

# Fit the model
heartLogReg.fit(X, np.ravel(y))

# Calculate the predicted probabilities
probs = heartLogReg.predict_proba(X)

print('Probabilities: {}'.format(probs))

# Calculate the predicted classes
classes = heartLogReg.predict(X)

print('Classes: {}'.format(classes))