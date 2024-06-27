# Import needed packages for regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Silence warning from sklearn
import warnings
warnings.filterwarnings('ignore')

diamonds = pd.read_csv('diamonds.csv')

#Input split value
splitSize = float(input())

# prompt: Map clarity values to numbers.
diamond_clarity_map = {
  "I1": 1,
  "SI2": 2,
  "SI1": 3,
  "VS2": 4,
  "VS1": 5,
  "VVS2": 6,
  "VVS1": 7,
  "IF": 8
}
diamonds['clarity_num'] = diamonds['clarity'].map(diamond_clarity_map)

# prompt: Map cut to numeric values to be used in model
diamond_cut_map = {
  "Fair": 1,
  "Good": 2,
  "Very Good": 3,
  "Premium": 4,
  "Ideal": 5
}
diamonds['cut_num'] = diamonds['cut'].map(diamond_cut_map)

# prompt: Generate mapping for color feature
diamond_color_map = {
  "J": 1,
  "I": 2,
  "H": 3,
  "G": 4,
  "F": 5,
  "E": 6,
  "D": 7
}
diamonds['color_num'] = diamonds['color'].map(diamond_color_map)

# Define the features and target variable
features = ['carat', 'cut_num', 'color_num', 'clarity_num', 'table']
target = 'price'

X = diamonds[features] 
y = diamonds[target]

poly_features = PolynomialFeatures(degree=3, include_bias=False)
# Transform the features
X_poly = poly_features.fit_transform(X)

# Split the data into training and test, and polynomial training and test
## Be sure to use variable splitSize for parameter test_size
## Be sure to use 777 for parameter random_state
### 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitSize, random_state=777)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=splitSize, random_state=777)

# Define the 2 ElasticNet regression models and the 2 Linear Regression Models
### Your Code Here
modelElastic = ElasticNet()
modelLinReg =  LinearRegression()
modelElasticPoly = ElasticNet()
modelLinRegPoly =  LinearRegression()

# Set the hyperparameters
modelElastic.set_params(alpha=0.5, l1_ratio=0.8)
modelElasticPoly.set_params(alpha=0.5, l1_ratio=0.8)

# Fit the models using X and X_poly for the regular and poly based models
## Be sure to train using your training set and polynomial training set.
modelElastic.fit(X_train,y_train)
modelLinReg.fit(X_train,y_train)
modelElasticPoly.fit(X_train_poly,y_train_poly)
modelLinRegPoly.fit(X_train_poly,y_train_poly)


# Print sum of the absolute value of the weights
print (f"{y_train.size=}")
sum_weights_modelLinReg = np.sum(np.abs(modelLinReg.coef_))
sum_weights_modelElastic = np.sum(np.abs(modelElastic.coef_))
sum_weights_modelElasticPoly = np.sum(np.abs(modelElasticPoly.coef_))
sum_weights_modelLinRegPoly = np.sum(np.abs(modelLinRegPoly.coef_))
print (f"{sum_weights_modelLinReg=:.3f}")
print (f"{sum_weights_modelElastic=:.3f}")
print (f"{sum_weights_modelElasticPoly=:.3f}")
print (f"{sum_weights_modelLinRegPoly=:.3f}")
# Print accuracy of each model
print(f"{modelLinReg.score(X_train, y_train)=:.3f}")
print(f"{modelLinReg.score(X_test, y_test)=:.3f}")
print(f"{modelElastic.score(X_train, y_train)=:.3f}")
print(f"{modelElastic.score(X_test, y_test)=:.3f}")
print(f"{modelElasticPoly.score(X_train_poly, y_train_poly)=:.3f}")
print(f"{modelElasticPoly.score(X_test_poly, y_test_poly)=:.3f}")
print(f"{modelLinRegPoly.score(X_train_poly, y_train_poly)=:.3f}")
print(f"{modelLinRegPoly.score(X_test_poly, y_test_poly)=:.3f}")