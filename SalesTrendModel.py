import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import cross_val_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import os
import pickle
from pandas.tseries.offsets import DateOffset

def read_data():
    df = pd.read_csv(os.path.dirname(__file__)+'/monthly-sales-train.csv')
    return df
df = read_data()

df['Month']=pd.to_datetime(df['Month'])
df_index = df.copy()
df.set_index('Month',inplace=True)

smodel =sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
fitted=smodel.fit(disp=-1)

filename = 'sarimax-model.pkl'

pickle.dump(fitted, open(filename, 'wb'))
