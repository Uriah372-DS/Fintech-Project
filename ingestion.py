import pandas as pd


def merge_stock_data(files_dict: dict[str, str], index_col: str = 'Gmt time') -> pd.DataFrame:
    """
    merge_stock_data reads the OHLC data from different stock files and merges them horizontally into a single dataframe.

    Parameters
    ----------
    files_dict : dict[str, str]
        A dictionary with column prefixes as keys and file paths as values.
    index_col : str, optional
        The name of the index column in all the files, by default 'Gmt time'.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    dfs = []
    for prefix, file_path in files_dict.items():
        df = pd.read_csv(file_path)
        df.set_index(index_col, inplace=True)
        df = df.add_prefix(f"{prefix}")
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=1)

    return merged_df


def extract() -> pd.DataFrame:
    """
    extract all the data from all the different data sources and create a pandas dataframe for later processing.

    Returns
    -------
    pd.DataFrame
        A single pandas dataframe that contains the raw data.
    """
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

    # TODO: Handle time discrepencies between different sources (Vladi).
    return df


if __name__ == '__main__':
    print(extract())
