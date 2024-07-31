# Import packages and functions
import pandas as pd

# Load the healthy_lifestyle dataset
healthy = pd.read_csv('healthy_lifestyle.csv')
healthy2 = pd.read_csv('healthy_lifestyle2.csv')

# Drop features from the original dataframe
healthy2.drop(axis=1, labels=['annual_hours_worked','pollution'], inplace=True)

# Count instances of duplicated values in the feature sunshine_hours
count = healthy.loc[:,'sunshine_hours'].duplicated().sum()


print(healthy.info())
print(count)
# Drop instances with missing values of sunshine_hours from the original dataframe
healthy2.dropna(subset=['sunshine_hours'], inplace=True)
print(healthy2.info())