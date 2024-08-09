import pandas as pd
from sklearn.preprocessing import StandardScaler
from ingestion import extract


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    pass


if __name__ == '__main__':
    df = extract()

    X = df.drop('Close', axis=1)
    y = df['Close']

    scaler = StandardScaler()

    # Fit and transform the data
    print(scaler.fit_transform(X))
