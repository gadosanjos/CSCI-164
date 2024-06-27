# Import needed packages for regression
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

# Silence warning from sklearn
import warnings
warnings.filterwarnings('ignore')

# Input feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv('diamonds.csv')

# Define input and output features
X = diamonds[['carat', 'table']]
y = diamonds['price']

# Initialize a k-nearest neighbors regression model using a Euclidean distance and k=12 
knnr = KNeighborsRegressor(n_neighbors=12, metric='minkowski', p=2)

# Fit the kNN regression model to the input and output features
knnrFit = knnr.fit(X,y)

# Create array with new carat and table values
Xnew = [[carat, table]]

# Predict the price of a diamond with the user-input carat and table values
prediction = knnrFit.predict(Xnew)
print('Predicted price is', np.round(prediction, 2))

# Find the distances and indices of the 12 nearest neighbors for the new instance
neighbors = knnrFit.kneighbors(Xnew)
print('Distances and indices of the 12 nearest neighbors are', neighbors)