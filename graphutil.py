import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def read_data():
    df = pd.read_csv(os.path.dirname(__file__)+'/monthly-sales-train.csv')
    return df

def graphForMonths():
    df= read_data()
    df['Month']=pd.to_datetime(df['Month'])
    df_index = df.copy()
    df.set_index('Month',inplace=True)
    plt.subplots(figsize=(10, 9))
    df.plot(color="#FF851B")
    my_path = os.path.dirname(__file__)+"/static/images"
    plt.title("Sales Trend from 2011 to 2019")
    plt.xlabel('Year')
    plt.ylabel('Sales') 
    my_file = 'graph.png'
    plt.savefig(os.path.join(my_path, my_file))

def graphForPrediction(future_df,m):
    plt.subplots(figsize=(9, 9))
    my_path = os.path.dirname(__file__)+"/static/images"
    future_df[['Sales', 'Predicted']].plot(figsize=(13, 9))
    plt.title("Sales Trend Prediction")
    plt.xlabel('Year')
    plt.ylabel('Sales') 
    my_file = 'graph_predicted'+m+'.png'
    plt.savefig(os.path.join(my_path, my_file))
    return "/static/images/"+my_file
