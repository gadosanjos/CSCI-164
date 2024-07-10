# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the hawks dataset
hawks = pd.read_csv('hawks.csv')

# Define input and output features
X = hawks[['Hallux', 'Weight']]
y = hawks[['Species']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a Gaussian naive Bayes model
NBModel = GaussianNB()
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
predHawk = NBModel.predict(X)

# Calculate the confusion matrix 
confMatrix = metrics.confusion_matrix(np.ravel(y), predHawk)

print('GaussianNB model\n', confMatrix)

# Calculate the accuracy
accuracy = metrics.accuracy_score(np.ravel(y), predHawk)

# Calculate kappa
kappa = metrics.cohen_kappa_score(np.ravel(y), predHawk)

print('GaussianNB model accuracy:', round(accuracy, 3))
print('GaussianNB model kappa:', round(kappa, 3))

# Calculate the precision
precision = metrics.precision_score(np.ravel(y), predHawk)

# Calculate the recall
recall = metrics.recall_score(np.ravel(y), predHawk)

print('GaussianNB model precision:', round(precision, 3))
print('GaussianNB model recall:', round(recall, 3))