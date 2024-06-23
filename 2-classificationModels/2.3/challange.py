# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the heart dataset
heart = pd.read_csv('heart.csv')

# Create a dataframe X containing chol and age
X = heart[['chol', 'age']]

# Output feature: target
y = heart['target']

# Initialize the model
NBModel = GaussianNB(priors=[0.2, 0.8])

# Fit the model
NBModel.fit(X, np.ravel(y))

# Calculate the predicted probabilities
probs = NBModel.predict_proba(X)

# Print predicted probabilities
print('Probabilities:', probs)