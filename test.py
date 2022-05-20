import pandas as pd
import joblib
import math
from sklearn.metrics import mean_squared_error as mse

def test(csv_path,model_path):
    dataframe = pd.read_csv(csv_path, index_col=0)
    dataframe['S_60'] = dataframe['Close'].rolling(window=60).mean()
    dataframe['Corr'] = dataframe['Close'].rolling(window=60).corr(dataframe['S_60'])
    dataframe['Open-Close'] = dataframe['Open'] - dataframe['Close'].shift(1)
    dataframe['Open-Open'] = dataframe['Open'] - dataframe['Open'].shift(1)
    df = dataframe.dropna().reset_index()
    del df["index"]
    df = pd.DataFrame(df)
    y = df.iloc[:, 4]
    y = y.iloc[1:]
    X = df.drop('Close', axis=1)
    X = X.iloc[:-1, :]
    estimator = joblib.load(model_path)
    return math.sqrt(mse(estimator.predict(X), y))
