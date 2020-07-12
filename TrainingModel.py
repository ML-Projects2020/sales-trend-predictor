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
import pmdarima as pm
import os
def read_data():
    df = pd.read_csv(os.path.dirname(__file__)+'/monthly-sales-train.csv')
    return df

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
        print("result[1]", result[1])
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

def SMAPE (forecast, actual):
    """Returns the Symmetric Mean Absolute Percentage Error between two Series"""
    masked_arr = ~((forecast==0)&(actual==0))
    diff = abs(forecast[masked_arr] - actual[masked_arr])
    avg = (abs(forecast[masked_arr]) + abs(actual[masked_arr]))/2
    
    print('SMAPE Error Score: ' + str(round(sum(diff/avg)/len(forecast) * 100, 2)) + ' %')
    
df = read_data()
sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

df['Month']=pd.to_datetime(df['Month'])
df_index = df.copy()
df.set_index('Month',inplace=True)

df.plot()
plt.show()
print("****************** adfuller_test(df['Sales']) ****************")
adfuller_test(df['Sales'])

# Checking Trend and Seasonality

print("****************** df rolling ****************")
df_rolling = df['Sales'].rolling(window=12).mean()
df_std = df['Sales'].rolling(window=12).std()
plt.plot(df['Sales'], color='blue',label='Original')
plt.plot(df_rolling, color='red',label='Original')
plt.plot(df_std, color='black',label='Original')
plt.show()

# Eliminating Trend and Seasonality

print("****************** Log rolling ****************")
df['LogSales'] = np.log(df['Sales'])
moving_avg = df['LogSales'].rolling(window=12).mean()
plt.plot(df['LogSales'])
plt.plot(moving_avg, color='red')
plt.title("LOG data frame")
plt.show()
adfuller_test(df['LogSales'])
print("****************** ewma rolling ****************")
df['ewma'] = df['Sales'].ewm(span=2).mean()
ewma_avg = df['ewma'].rolling(window=12).mean()
plt.plot(df['ewma'])
plt.plot(ewma_avg, color='red')
plt.title("ewma data frame")
plt.show()
adfuller_test(df['ewma'])

print("****************** adfuller_test(df['Seasonal First Difference']) ****************")
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)
diff_sales_avg = df['Seasonal First Difference'].rolling(window=12).mean()
plt.plot(diff_sales_avg, color="red")
df['Seasonal First Difference'].plot()
plt.title("Seasonal First Difference")
plt.show()
adfuller_test(df['Seasonal First Difference'].dropna())


print(" ************** Differenceing on log Sales & adfuller test**************")
df['Seasonal First Difference LogSales'] =  - df['LogSales'].shift()
diff_logsales_avg = df['Seasonal First Difference LogSales'].rolling(window=12).mean()
plt.plot(diff_logsales_avg, color="red")
df['Seasonal First Difference LogSales'].plot()
plt.title("Log First Difference")
plt.show()
adfuller_test(df['Seasonal First Difference LogSales'].dropna())

model = ARIMA(df['Sales'], order=(1,1,1))  
model_fit = model.fit()
print(model_fit.summary())

# Actual vs Fitted
# model_fit.plot_predict(dynamic=False)
# plt.title("Actual vs Fitted")
# plt.show()


df['forecast']=model_fit.predict(start=98,end=109,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))
plt.title("Sales vs Forcast")
plt.show()


smodel =sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
fitted=smodel.fit(disp=-1)
predictions = fitted.predict(start=96,end=107,dynamic=True)
df['forecast']= predictions
df[['Sales','forecast']].plot(figsize=(12,8))
plt.show()

print("************************** SMAPE *********************** GET df between two index 98 109, added df_index")
smape_df = df_index[(df_index.index >= 96) & (df_index.index <= 107)]
print(smape_df)
smape_df['forecast']= np.array(predictions)
SMAPE(smape_df['forecast'], smape_df['Sales'])


from pandas.tseries.offsets import DateOffset
print("Dateoffset")
print(df.index[-1])
future_dates=[df.index[-1]+ DateOffset(months=x+1)for x in range(0,12)]
future_datest_df=pd.DataFrame(index=future_dates[0:],columns=df.columns)
print(future_datest_df)

future_df=pd.concat([df,future_datest_df])
future_df['forecast'] = fitted.predict(start = 107, end = 121, dynamic= True).round(decimals=0)
print(future_df.tail(12) )

future_df[['Sales', 'forecast']].plot(figsize=(12, 8))
plt.show()


