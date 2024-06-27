# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[['economy_gdp_per_capita', 'trust_government_corruption']]
y = happiness[['happiness_score']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize an elastic net regression model
#The parameter alpha controls the strength of the regularization, 
# with alpha=0 being the same as least squares linear regression. 
# The parameter l1_ratio controls the ratio between L1 and L2. 
# When l1_ratio=0, the function becomes a ridge regression function. When l1_ratio=1, the function becomes a LASSO function.
eNetModel = ElasticNet(alpha=1, l1_ratio=.5)

# Fit the model 
eNetModel.fit(X,y)

# Estimated intercept weight
print(eNetModel.intercept_)

# Estimated weights for input features
print(eNetModel.coef_)

# Predict the happiness score for a country with economy_gdp_per_capita = -0.49 and trust_government_corruption = -0.96
print(eNetModel.predict([[-0.49,-0.96]]))