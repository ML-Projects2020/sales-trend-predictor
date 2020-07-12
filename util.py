import pandas as pd

def addMonthYear(dataframe):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['month'] = dataframe['date'].apply(lambda x: x.month)
    dataframe['year'] = dataframe['date'].apply(lambda x: x.year)
    return dataframe

def divideTheDate(dataframe):
    dataframe['quarter'] = dataframe['date'].apply(lambda x: x.quarter)
    dataframe['day_of_week'] = dataframe['date'].apply(lambda x: x.dayofweek)
    dataframe['week_of_year'] = dataframe['date'].apply(lambda x: x.weekofyear)
    dataframe['day_of_year'] = dataframe['date'].apply(lambda x: x.dayofyear)
    dataframe['Is_Mon'] = (dataframe.day_of_week == 0) *1
    dataframe['Is_Tue'] = (dataframe.day_of_week == 1) *1
    dataframe['Is_Wed'] = (dataframe.day_of_week == 2) *1
    dataframe['Is_Thu'] = (dataframe.day_of_week == 3) *1
    dataframe['Is_Fri'] = (dataframe.day_of_week == 4) *1
    dataframe['Is_Sat'] = (dataframe.day_of_week == 5) *1
    dataframe['Is_Sun'] = (dataframe.day_of_week == 6) *1
    dataframe['Is_wknd'] = dataframe.day_of_week // 4
    return dataframe

def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','day_of_week'])['sales'].transform('mean')
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean')
    x['day_of_year_avg']=x.groupby(['item','store','day_of_year'])['sales'].transform('mean')
    x['week_of_year_avg']=x.groupby(['item','store','week_of_year'])['sales'].transform('mean')
    x['year_avg'] = x.groupby(['item','store','year'])['sales'].transform('mean')
    return x

def daily_avg(dataframe):
    daily_avg = dataframe.groupby(['item','store','day_of_week'])['sales'].mean().reset_index()
    return  daily_avg

def monthly_avg(dataframe):
    monthly_avg = dataframe.groupby(['item','store','month'])['sales'].mean().reset_index()
    return  monthly_avg

def day_of_year_avg(dataframe):
    day_of_year_avg = dataframe.groupby(['item','store','day_of_year'])['sales'].mean().reset_index()
    return  day_of_year_avg

def week_of_year_avg(dataframe):
    week_of_year_avg = dataframe.groupby(['item','store','week_of_year'])['sales'].mean().reset_index()
    return  week_of_year_avg

def year_avg(dataframe):
    year_avg = dataframe.groupby(['item','store','year'])['sales'].mean().reset_index()
    return  year_avg

def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x