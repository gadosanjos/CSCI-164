import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

diamonds = pd.read_csv('diamonds.csv')

diamonds = diamonds.sample(n=800, random_state=10)

# Create a dataframe X containing all the features except cut, color, clarity, and price
X = diamonds[['carat', 'depth', 'table', 'x', 'y','z']]
# Create a dataframe y containing the feature price
y = diamonds[['price']]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=123)

# Initialize a multilayer perceptron regressor with two hidden layers with 50 nodes each, 
# the 'identity' activation function, max_iter=500, and random_state=123
mlpModel =MLPRegressor(hidden_layer_sizes=(50,50), activation='identity', max_iter=500, random_state=123)

# Fit the model
mlpModel.fit(Xtrain, np.ravel(ytrain))

# Print the price predictions and actual prices for the first five rows of the dataframe
pred = mlpModel.predict(Xtrain)
print("Price predictions:", pred[0:5])
print("Actual prices: \n", ytrain[0:5])

# Print the R-squared score for the training data
print("Score for the training data: ", round(mlpModel.score(Xtrain, ytrain), 4))
# Print the R-squared score for the testing data
print("Score for the testing data: ", round(mlpModel.score(Xtest, ytest), 4))