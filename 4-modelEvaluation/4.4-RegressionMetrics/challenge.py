# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[['freedom', 'health_life_expectancy']].values.reshape(-1, 2)
y = happiness[['happiness_score']].values.reshape(-1, 1)

# Scale the input features
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

# Initialize and fit a Gaussian naive Bayes model
enModel = ElasticNet(alpha=1, l1_ratio=0.5)
enModel.fit(Xscaled, y)

# Calculate the predictions for each instance in X
predEN = enModel.predict(Xscaled)

# Calculate the mean absolute error
enMAE = metrics.mean_absolute_error(y, predEN)

# Calculate the mean squared error 
mseEN = metrics.mean_squared_error(y, predEN)

# Calculate the R-squared 
happinessR2 = metrics.r2_score(y, predEN)

# Calculate the mean absolute percentage error
mapeHappiness = metrics.mean_absolute_percentage_error(y, predEN)

print("Mean absolute error:", enMAE)
print("Mean squared error:", mseEN)
print("R-squared:", happinessR2)
print("Mean absolute percentage error:", mapeHappiness)