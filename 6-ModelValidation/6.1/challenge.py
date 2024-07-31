# Import packages and functions
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['ConvexArea', 'EquivDiameter']]
y = beans[['Class']]

# Set aside 40% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=27, stratify=y)

# Split training again into 50% training and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1/(1-0.4), random_state=27, stratify=y_train)

# Print split sizes and test dataset
print('original dataset:', len(beans), 
    '\ntrain_data:', len(X_train), 
    '\nvalidation_data:', len(X_val), 
    '\ntest_data:', len(X_test),
    '\n', X_test
)

"""  
# Set aside 40% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=156, stratify=y)

# Split training again into 50% training and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1/(1-0.4), random_state=156, stratify=y_train)

"""

""" 
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate

# Load the dry beans dataset
beans = pd.read_csv('Dry_Bean_Data.csv')

X = beans[['Eccentricity', 'roundness']]
y = beans[['Class']]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
beansModel = LinearDiscriminantAnalysis(n_components=2, store_covariance=True)

# Fit linear discriminant analysis model with 4-fold cross-validation
beanResults = cross_validate(beansModel, X, np.ravel(y), cv=4)

# Print test score
print('Test score:', beanResults['test_score'])

"""
