# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:15:58 2021

@author: jeti8
"""
import pandas as pd
from fbprophet import Prophet
import numpy as np
import tensorflow as tf
import pathlib
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
import datetime
from pathlib import Path
import os
from numpy import array

        
def annotate_points(x_data, y_data, ax, interval_between = 10):
    for i, (x, y) in enumerate(zip(x_data, y_data)):   
        if (i % interval_between == 0):
            if not math.isnan(y):       
                label ="{:.2f}".format(y)               
            
                ax.annotate(label, # this is the text
                             (x,y), # these are the coordinates to position the label
                             textcoords="offset points", # how to position the text
                             xytext=(0, 2), # distance from text to points (x,y)
                             ha='left') # 

def save_plot(directory):
    datatime = datetime.datetime.today()
    datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)
    filename = directory + '/' + currency + '-' + datestr + '.png';
        
    if os.path.exists(filename): os.remove(filename)
        
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)

datatime = datetime.datetime.today()
datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)

current_abs_path = str(pathlib.Path().resolve())
save_plot_dir = current_abs_path + '/predictions/prediction-prophet/' + str(datestr)

data = pd.read_json('../../cryptocurrency_rates_history.json')
currency_pairs = data[['full_date', 'pair', 'ask']]
currencies = currency_pairs['pair'].drop_duplicates()
scaler = MinMaxScaler()

datatime = datetime.datetime.today()
datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)
save_plot_dir = current_abs_path + '/predictions/prediction-prophet/' + str(datestr)

#24 * 30 = 720
# 24 * 90 = 2160
int_hours_pred = 2160

for currency in currencies:
    if currency == 'ETH-EUR':    
        currency_pairs_eth = currency_pairs['pair'] == currency
        data_filtered = currency_pairs[currency_pairs_eth]
        currency_dates = pd.to_datetime(data_filtered['full_date'])
        currency_dates[:] = np.reshape(currency_dates, (-1))
        
        
        data_filtered['full_date'].iloc[:] = pd.to_datetime(data_filtered['full_date'], format='%Y-%m-%d', utc=True, errors='coerce').dt.strftime("%Y-%m-%d")

        # reshape in 2D
        close_data = data_filtered[['full_date', 'ask']].rename(columns={'full_date':'ds', 'ask': 'y'})
    

        m = Prophet()
        m.fit(close_data)
        
        future = m.make_future_dataframe(periods=int_hours_pred, freq='H')
        future.tail()
        
        forecast = m.predict(future)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        today = datetime.datetime.today()
        forecast_endday = today + relativedelta(hours=int_hours_pred)
        forecast_startdate = today


        fig1 = m.plot(forecast, figsize=(120, 120))
        
              
        fig1.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
        
        fig1.gca().set_xlim([forecast_startdate, forecast_endday])
        fig1.gca().set_yticks(np.arange(np.min(forecast.yhat[-int_hours_pred:]) - 1, np.max(forecast.yhat[-int_hours_pred:]) + 1, 10))   
        fig1.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        annotate_points(forecast.ds, forecast.yhat, fig1.gca(), 30)
        
        save_plot(save_plot_dir)
        


        
    





















