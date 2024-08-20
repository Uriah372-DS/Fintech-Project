import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam,SGD

def build_and_compile_model(input_shape, num_layers=2, hidden_dims=[150, 100], dropout_rate=0,
                            dense_units=[50, 25], optimizer='adam', loss='mean_squared_error',lr=1e-3):
    model = Sequential()

    # Adding LSTM layers with specified hidden dimensions and dropout rates
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)  # return_sequences=True for all layers except the last LSTM layer
        model.add(LSTM(hidden_dims[i], return_sequences=return_sequences, input_shape=input_shape if i == 0 else None))
        model.add(Dropout(dropout_rate))

    # Adding Dense layers
    for units in dense_units:
        model.add(Dense(units))

    # Output layer
    model.add(Dense(1))
    if optimizer=='adam':
        optimizer = Adam(learning_rate=lr)
    if optimizer=='SGD':
        optimizer = SGD(learning_rate=lr)
    # Compile model
    model.compile(optimizer=optimizer, loss=loss)

    return model