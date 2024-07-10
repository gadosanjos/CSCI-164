# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load the breast cancer dataset
cancer = pd.read_csv('WisconsinBreastCancerDatabase.csv')

# Select input and output features
X = cancer.drop(columns=['ID', 'Diagnosis', 'Texture worst'])
y = cancer[['Diagnosis']]

# Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Initialize a multilayer perceptron classifier with default parameters and
# random_state set to 67
WisconsinModel = MLPClassifier(hidden_layer_sizes=(200), activation='identity',random_state=82)

# Fit multilayer perceptron classifier to X and y
WisconsinModel.fit(X, np.ravel(y))

# Print R-squared score for training data
trainScore = WisconsinModel.score(X_train, y_train)
print(trainScore)

# Print R-squared score for testing data
testScore = WisconsinModel.score(X_test, y_test)
print(testScore)

# Neural network structure
print('Number of total layers:', WisconsinModel.n_layers_)
print('Number of hidden layers:', WisconsinModel.n_layers_ - 2)
print('Number of outputs:', WisconsinModel.n_outputs_)