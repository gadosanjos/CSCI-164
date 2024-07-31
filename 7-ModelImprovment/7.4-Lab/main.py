"""  
The homes.csv dataset contains information for houses sold in King County, Washington in 2014. Features include sales price, square footage, number of bedrooms and bathrooms, and the number of floors. The priceFloor dataframe contains the features Price and Floor. The school dataframe contains the feature School.

Define a standardization scaler to transform values and apply the scaler to the priceFloor data.
Define a normalization scaler to transform values and apply the scaler to the priceFloor data.
Define an ordinal encoder using OrdinalEncoder(). Apply the ordinal encoder to the school data. Add the encoded labels as a column labeled encoding to the school dataframe.
Create and fit a discretizer with equal weights and 3 bins to the Floor feature from the priceFloor data. Reshape the feature to an array with dimensions (76,1).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

homes = pd.read_csv('homes.csv')
priceFloor = homes[['Price', 'Floor']]
school = homes[['School']]

# Define a standardization scaler to transform values
transform = StandardScaler()

# Apply scaler to the priceFloor data
scaled = transform.fit_transform(priceFloor)

homes_standardized = pd.DataFrame(scaled, columns=['Price','Floor'])
print('Standardized data: \n', homes_standardized)

# Define a normalization scaler to transform values
normalization =  MinMaxScaler()

# Apply scaler to the priceFloor data
normalized = normalization.fit_transform(priceFloor)

homes_normalized = pd.DataFrame(normalized, columns=['Price','Floor'])
print('Normalized data: \n', homes_normalized)

# Define the OrdinalEncoder() function
ordinal_encoder = OrdinalEncoder()
# Create a dataframe of the ordinal encoder function fit to the school data, with the column labeled encoding
labels = pd.DataFrame(ordinal_encoder.fit_transform(school[['School']]),columns=['encoding'])

# Join the new column to the school data
school_encoded = school.join(labels)

print('Encoded data: \n', school_encoded)

# Create a discretizer with equal weights and 3 bins
discretizer_eqwidth = KBinsDiscretizer(n_bins=3, strategy='uniform')

# Fit the discretizer to the Floor feature from the priceFloor data. 
# Reshape the feature to an array with dimensions (76,1).
discretizer_eqwidth.fit(np.reshape(homes[['Floor']],(76,1)))

print('Bin widths: \n', discretizer_eqwidth.bin_edges_[0])