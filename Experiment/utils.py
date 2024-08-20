from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def split_time_series(df, train_size=0.8, val_size=0.1):
    total_size = len(df)
    train_end = int(total_size * train_size)
    val_end = train_end + int(total_size * val_size)
    train = df.iloc[:train_end]
    validation = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, validation, test

def transform_and_scale(df_train, df_val, df_test):
    scaler_train = StandardScaler()
    train_scaled = pd.DataFrame(scaler_train.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
    val_scaled = pd.DataFrame(scaler_train.transform(df_val), columns=df_val.columns, index=df_val.index)
    test_scaled = pd.DataFrame(scaler_train.transform(df_test), columns=df_test.columns, index=df_test.index)
    return train_scaled, val_scaled, test_scaled

def prepare_sequences(df: pd.DataFrame, target_column="Close", sequence_length=30):
    """
    Prepares sequences from the DataFrame for LSTM input.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the features and target.
    target_column : str
        The name of the target column.
    sequence_length : int
        The length of the sequences.

    Returns
    -------
    tuple
        The sequences for X (features) and y (target).
    """
    feature_columns = df.columns[df.columns != target_column]
    features = df[feature_columns].values
    target = df[target_column].values

    X = []
    y = []

    for i in range(len(df) - sequence_length):
        end_ix = i + sequence_length
        X.append(features[i:end_ix])
        y.append(target[end_ix])

    return np.array(X), np.array(y)