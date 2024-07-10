import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Set the backend using tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Suppress tensorflow INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# The backend must be set before importing keras, not after
import keras
keras.utils.set_random_seed(812)

df = pd.read_csv('diamonds.csv')
diamond_sample = df.sample(1000, random_state=12)

X = diamond_sample.drop(columns=['cut', 'color', 'clarity', 'price'])
y = diamond_sample[['price']]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

# Define the model structure using keras.Sequential. The input layer has shape=(6, ), hidden layer 1 has
# 256 nodes and relu activation, hidden layer 2 had 128 nodes and linear activation, hidden layer 3 has 
# 64 nodes and linear activation, and the output layer has 1 node (for regression) and linear activation

model = keras.Sequential(
    [
        # Input layer
        keras.layers.Input(shape=(6, )),
        # Hidden layer 1 = 256 nodes, linear activation
        keras.layers.Dense(256, activation='relu'),
        # Hidden layer 2: 128 nodes, linear activation
        keras.layers.Dense(128, activation='linear'),
         # Hidden layer 3: 64 nodes, linear activation
        keras.layers.Dense(64, activation='linear'),
        # Output layer: 1 node
        keras.layers.Dense(1, activation='linear'),
    ]
)

print(model.summary())


# Specify training choices using the compile method, with optimizer='Adam', loss='MeanSquaredError',
# metrics='mse'
model.compile(
    optimizer='Adam',  # Optimizer
    # Loss function to minimize
    loss='MeanSquaredError',
    # List of metrics to monitor
    metrics=['mse'],
)

# Train the model with a batch size of 100, 5 epochs, validation_split=0.1, and verbose=0
model.fit(Xtrain, ytrain, batch_size=100, epochs=5, validation_split=0.1, verbose=0)

predictions = model.predict(Xtest[:3], verbose=0)
print('Predictions:', predictions.round(3))
print('Actual values:', ytest[:3])