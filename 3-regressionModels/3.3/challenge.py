# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Load the world happiness dataset
happiness = pd.read_csv('world_happiness_2017.csv')

# Define input and output features
X = happiness['generosity'].values.reshape(-1, 1)
y = happiness[['happiness_score']]

# Initialize kNN regression model with k = 19
happinessModel =  KNeighborsRegressor(n_neighbors=15,metric='cosine')

# Fit the model 
happinessModel.fit(X,y)

# New instance
Xnew = np.array([[0.28]])

# Find the indices and distances of the 15 nearest neighbors of the new instance
nearestNeighbors = happinessModel.kneighbors(Xnew)
print(nearestNeighbors)

# Return the data frame instances of the 15 nearest neighbors
nearestNeighborsDf = happiness.iloc[happinessModel.kneighbors(Xnew)[1][0]]
print(nearestNeighborsDf)