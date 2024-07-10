# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

# Load the forest fires dataset
fires = pd.read_csv('forestfires.csv')

# Select input and output features
Xtrain = fires.drop(columns=['X', 'Y', 'month', 'day', 'area', 'DC'])
ytrain = fires[['area']]

# Initialize a multilayer perceptron regressor with mini-batch gradient descent and random_state=24
forestFireModel = MLPRegressor(solver='sgd',batch_size=90,random_state=24)

# Fit the model to the training data
forestFireModel.fit(Xtrain, np.ravel(ytrain))

# Print the minimum loss reached
print(forestFireModel.score(Xtrain, ytrain))