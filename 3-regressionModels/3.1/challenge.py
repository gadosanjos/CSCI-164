# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[['freedom', 'trust_government_corruption']].values.reshape(-1, 2)
y = happiness[['happiness_score']].values.reshape(-1, 1)

# Initialize a simple linear regression model
happinessSLR = LinearRegression()

# Fit a simple linear regression model
happinessSLR.fit(X,y)

# Estimated intercept weight
print(happinessSLR.intercept_)

# Estimated weight for generosity feature
print(happinessSLR.coef_)

# Predict the happiness score for a country with freedom = 0.3 and trust_government_corruption = 0.5
print(happinessSLR.predict([[0.3,0.5]]))