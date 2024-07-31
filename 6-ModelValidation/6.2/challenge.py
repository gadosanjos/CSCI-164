# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Eccentricity', 'ConvexArea']]
y = beans[['Class']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a set of 9 cross-validation folds
kf = KFold(n_splits=9, shuffle=True, random_state=197)

# Initialize the linear discriminant analysis model with two components
LDAmodel = LinearDiscriminantAnalysis(n_components=2)

# Fit linear discriminant analysis model with cross-validation
modelLDAResults = cross_validate(LDAmodel, X, np.ravel(y), cv=kf)

LDABeanScores = modelLDAResults['test_score']

# View accuracy for each fold
print('Linear discriminant analysis scores:', LDABeanScores.round(3))

# Calculate descriptive statistics
print('Mean:', LDABeanScores.mean().round(3))
print('SD:', LDABeanScores.std().round(3))

# Initialize the Gaussian naive Bayes model
beansNBModel = GaussianNB()

# Fit Gaussian naive Bayes model with 9-fold cross-validation
NBresults = cross_validate(beansNBModel, X, np.ravel(y), cv=kf)

NBBeanScores = NBresults['test_score']

# View accuracy for each fold
print('Gaussian naive Bayes scores:', NBBeanScores.round(3))

# Calculate descriptive statistics
print('Mean:', NBBeanScores.mean().round(3))
print('SD:', NBBeanScores.std().round(3))

"""  
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Compactness', 'Solidity']]
y = beans[['Class']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a set of 8 cross-validation folds
kf = KFold(n_splits=8, random_state=131, shuffle=True)

# Initialize the linear discriminant analysis model with two components
modelLDA = LinearDiscriminantAnalysis(n_components=2)

# Fit linear discriminant analysis model with cross-validation
LDAresults = cross_validate(modelLDA, X, np.ravel(y), cv=kf)

modelLDAScores = LDAresults['test_score']

# View accuracy for each fold
print('Linear discriminant analysis scores:', modelLDAScores.round(3))

# Calculate descriptive statistics
print('Mean:', modelLDAScores.mean().round(3))
print('SD:', modelLDAScores.std().round(3))

# Initialize the Gaussian naive Bayes model
beansNB = GaussianNB()

# Fit Gaussian naive Bayes model with cross-validation
NBBeanResults = cross_validate(beansNB, X, np.ravel(y), cv=kf)

NBBeanScores = NBBeanResults['test_score']

# View accuracy for each fold
print('Gaussian naive Bayes scores:', NBBeanScores.round(3))

# Calculate descriptive statistics
print('Mean:', NBBeanScores.mean().round(3))
print('SD:', NBBeanScores.std().round(3))
"""