# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# Load nbaallelo_log.csv into a dataframe
NBA = NBA = pd.read_csv('nbaallelo_log.csv').dropna()

# Create binary feature for game_result with 0 for L and 1 for W
NBA['win'] = NBA['game_result'].replace(to_replace = ['L','W'],value = [int(0), int(1)])

# Store relevant columns as variables
X = NBA[['elo_i']]
y = NBA[['win']]

# Initialize and fit the logistic model using the LogisticRegression() function
logisticModel = LogisticRegression()
logisticModel.fit(X, np.ravel(y))

# Print the weights for the fitted model
print('w1:', logisticModel.coef_)

# Print the intercept of the fitted model
print('w0:', logisticModel.intercept_)

# Find the proportion of instances correctly classified
score = logisticModel.score(X, np.ravel(y))
print(round(score, 3))