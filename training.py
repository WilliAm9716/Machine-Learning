import math
import matplotlib.pyplot
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error as mse
import joblib

# data preparation
dataframe = pd.read_csv('Stock_price_training_data.csv', index_col=0)
dataframe['S_60'] = dataframe['Close'].rolling(window=60).mean()
dataframe['Corr'] = dataframe['Close'].rolling(window=60).corr(dataframe['S_60'])
dataframe['Open-Close'] = dataframe['Open'] - dataframe['Close'].shift(1)
dataframe['Open-Open'] = dataframe['Open'] - dataframe['Open'].shift(1)
print(dataframe.isnull().sum())
df = dataframe.dropna().reset_index()
del df["index"]
print(df.head())

# split dataset
df = pd.DataFrame(df)
y = df.iloc[:, 4]
y = y.iloc[1:]
X = df.drop('Close', axis=1)
X = X.iloc[:-1, :]
split = int(0.8*len(df))
X_train, X_validation = X[:split], X[split:]
y_train, y_validation = y[:split], y[split:]

# regression
para = np.linspace(0.01, 5, 20)
y_pred_train = np.zeros(20)
y_pred_validation = np.zeros(20)
m = 0
for i in para:
    ridge_reg= Ridge(i)
    ridge_reg.fit(X_train, y_train)
    y_pred_train[m] = math.sqrt(mse(ridge_reg.predict(X_train), y_train))
    y_pred_validation[m] = math.sqrt(mse(ridge_reg.predict(X_validation), y_validation))
    m += 1

# plotting
#pyplot.plot(para, y_pred_train, 'b')
pyplot.plot(para, y_pred_validation, 'r')
pyplot.xlabel('$\lambda$'); pyplot.ylabel('RMSE')
matplotlib.pyplot.show()

# find the most suitable lambda whose RMSE is the smallest
print(y_pred_validation)
print('The most suitable lambda = ', para[6])

# save the model
ridge_reg= Ridge(1.59)
ridge_reg.fit(X_train, y_train)
print(pd.DataFrame(zip(X.columns, np.transpose(ridge_reg.coef_))))
joblib.dump(ridge_reg,'./model.pkl')