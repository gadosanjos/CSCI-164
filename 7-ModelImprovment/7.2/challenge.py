# Import packages and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load the taxis dataset
taxis = pd.read_csv('taxis.csv')

# Subset tip and total
taxis = taxis[['tip', 'total']]

# Initialize a standardization scaler to transform values
scaled = StandardScaler()

# Apply scaler to taxis
data_stand = scaled.fit_transform(taxis)

# Initialize a normalization scaler to transform values
norm = MinMaxScaler()

# Apply scaler to taxis
norm_data = norm.fit_transform(taxis)

print(data_stand)
print(norm_data)