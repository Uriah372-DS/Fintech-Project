import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ingestion import extract
from indicators import weighted_moving_average, bollinger_bands, rsi


def transform(df: pd.DataFrame) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
    """
    transform the raw data and creates the final features for the model (including indicators).

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe, as returned by the `extract` function.

    Returns
    -------
    pd.DataFrame | tuple[np.ndarray, np.ndarray]
        The dataframe after all the transformations,
        preprocessing and added features that are required to load into the forecasting model.
    """
    # shift to treat Closing price as label
    df['Close'] = df['Close'].shift(-1)

    # Calculate the moving average, standard deviation, and Bollinger Bands
    df = weighted_moving_average(df=df, ohlc_column="Close", window_size=20)
    df = bollinger_bands(df=df, ohlc_column="Close")
    df = rsi(df=df, ohlc_column="Close", window_size=14)

    X = df.drop('Close', axis=1)
    y = df['Close']

    # TODO: Find a good scaling algorithm for the data. StandardScaler is basic and not justified theoretically.
    scaler = StandardScaler()

    # Fit and transform the data
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


if __name__ == '__main__':
    raw_data = extract()
    preprocessed_data = transform(df=raw_data)
    print(preprocessed_data)
