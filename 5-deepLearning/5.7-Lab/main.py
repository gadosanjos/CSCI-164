import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

diamonds = pd.read_csv('diamonds.csv')
diamond_sample = diamonds.sample(1000, random_state=123)

X = diamond_sample.drop(columns=['cut', 'color', 'clarity', 'price'])
y = diamond_sample[['price']]

# Split data into train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=123)

# Define a standardization scaler to transform values
transform = StandardScaler()

# Apply scaler
Xtrain = transform(Xtrain)
Xtest = transform(Xtest)

# Initialize a multilayer perceptron regressor with random_state=42, three hidden layers of 50 nodes each, 
# an adaptive learning rate of 0.01, a batch size of 100, and a maximum of 300 iterations
mlpDiamond = MLPRegressor(hidden_layer_sizes=(50,50,50), learning_rate_init=0.01, activation='identity', max_iter=300, random_state=42, batch_size=100)

# Fit the model to the training data
mlpDiamond.fit(Xtrain, np.ravel(ytrain))

# Print the R-squared score for the training data
print("Score for the training data: ", round(mlpDiamond.score(Xtrain, ytrain), 4))
# Print the R-squared score for the testing data
print("Score for the testing data: ", round(mlpDiamond.score(Xtest, ytest), 4))