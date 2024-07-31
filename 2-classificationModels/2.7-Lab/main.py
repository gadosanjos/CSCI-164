# Import the necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from mlxtend.plotting import plot_decision_regions

# Load the dataset
skySurvey = pd.read_csv('SDSS.csv').dropna()

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift','u_g']]

# Create dataframe y with feature class
y = skySurvey[['class']]

# Initialize a Gaussian naive Bayes model
skySurveyNBModel = GaussianNB()

# Fit the model
skySurveyNBModel.fit(X, np.ravel(y))

# Calculate the proportion of instances correctly classified
score = skySurveyNBModel.score(X, np.ravel(y))

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)