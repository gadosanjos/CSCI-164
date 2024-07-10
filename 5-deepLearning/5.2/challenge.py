# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
cancer = pd.read_csv('WisconsinBreastCancerDatabase.csv')

# Select input and output features
X = cancer.drop(columns=['ID', 'Diagnosis', 'Radius worst'])
y = cancer[['Diagnosis']]

# Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Standardize the input feature values in X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize a multilayer perceptron classifier with learning_rate_init=0.1, max_iter=250, and random_state=54
modelPerceptron = MLPClassifier(learning_rate_init=0.1, max_iter=250, random_state=54)

# Initialize a multilayer perceptron classifier with the specified parameters
modelPerceptron2 = MLPClassifier(learning_rate_init=0.01, max_iter=350, random_state=39)

# Fit multilayer perceptron classifier to X_train and y_train
modelPerceptron.fit(X_train, np.ravel(y_train))

# Fit multilayer perceptron classifier to X_train and y_train
modelPerceptron2.fit(X, np.ravel(y))

# Print the minimum loss reached
bestLoss = modelPerceptron.best_loss_
print(bestLoss)

# Neural network structure
print("Number of total layers:", modelPerceptron2.n_layers_)
print("Number of hidden layers:", modelPerceptron2.n_layers_ - 2)
print("Number of outputs:", modelPerceptron2.n_outputs_)

# Print the minimum loss reached
bestLoss = modelPerceptron2.best_loss_
print(bestLoss)