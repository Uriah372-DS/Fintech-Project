import pandas as pd
from indicators import merge_stock_data, weighted_moving_average, bollinger_bands, rsi


def extract() -> pd.DataFrame:
    files_dict = {
        "": r"data\GBPJPY_Candlestick_1_D_BID_01.01.2014-03.08.2024.csv",
        "barclays_": r"data\barclays_stock.csv",
        "mitsui_": r"data\mitsui_stock.csv",
        # "hitachi_": r"data\hitachi_stock.csv",
        "nissan_": r"data\nissan_stock.csv",
        "honda_": r"data\honda_stock.csv",
        "sony_": r"data\sony_stock.csv"
    }

    df = merge_stock_data(files_dict)

    # shift to treat Closing price as label
    df['Close'] = df['Close'].shift(-1)

    # Calculate the moving average, standard deviation, and Bollinger Bands
    df = weighted_moving_average(df=df, ohlc_column="Close", window_size=20)
    df = bollinger_bands(df=df, ohlc_column="Close")
    df = rsi(df=df, ohlc_column="Close", window_size=14)
    return df


if __name__ == '__main__':
    print(extract())
