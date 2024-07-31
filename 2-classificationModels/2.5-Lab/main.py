# Import needed packages for classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import packages for evaluation
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
skySurvey = pd.read_csv('SDSS.csv')

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift', 'u_g']]

# Create dataframe y with feature class
y = skySurvey[['class']]

np.random.seed(42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize model with k=3
skySurveyKnn = KNeighborsClassifier(n_neighbors=3)

# Fit model using X_train and y_train
skySurveyKnn.fit(X_train, np.ravel(y_train))

# Find the predicted classes for X_test
y_pred = skySurveyKnn.predict(X_test)

# Calculate accuracy score
score = skySurveyKnn.score(X_test, np.ravel(y_test))

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)