# Importing essential libraries
from flask import Flask, render_template, request, Response
import pickle
import numpy as np
from datetime import date
import locale
import graphutil as util
from pandas.tseries.offsets import DateOffset
import pandas as pd
import os
from pathlib import Path

model = pickle.load(open("sarimax-model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    util.graphForMonths()
    return render_template('index.html', linePlotPath="\static\images\graph.png", samplePath="\static\images\_train_data.PNG")
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        months = str(request.form['months'])
        
        df = util.read_data()
        df['Month']=pd.to_datetime(df['Month'])
        df.set_index('Month',inplace=True)
        
        future_dates=[df.index[-1]+ DateOffset(months=x+1)for x in range(0,int(months))]
        future_datest_df=pd.DataFrame(index=future_dates[0:],columns=df.columns)

        endNumber = 107 + int(months)
        predictions = model.predict(start = 107, end = endNumber, dynamic= True).round()
        future_df=pd.concat([df,future_datest_df])

        future_df['Predicted'] = predictions
        future_datest_df['Sales'] = predictions
        
        future_datest_df.reset_index(level=0, inplace=True)
        future_datest_df.rename(columns={'index':'Month'}, inplace=True)
        future_datest_df['Month'] = future_datest_df['Month'].dt.strftime('%m /%Y')

        file_path = os.path.dirname(__file__)+'/predicted-monthly-sales.csv'
        
        data_folder = Path("static/csv/")
        file_to_open = data_folder / "predicted-monthly-sales.csv"

        future_datest_df.to_csv(file_to_open)
        print("file_path", file_to_open)
        return render_template('result.html', path=util.graphForPrediction(future_df,months), path_to_file = file_to_open)

@app.route('/getCsv') # this is a job for GET, not POST
def plot_csv():
    fp = open(os.path.dirname(__file__)+'/predicted-monthly-sales.csv')
    csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=predicted-monthly-sales.csv"})
    
if __name__ == '__main__':
	app.run(debug=True)