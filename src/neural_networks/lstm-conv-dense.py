# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:54:20 2021

@author: rodion kovalenko
"""

# LSTM for international airline passengers problem with regression framing

import pandas as pd
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
import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import os
import keras
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

def predict(num_prediction, model, n_seq, n_steps, n_features, look_back, close_data):
    # prediction_list = close_data_plot[-look_back:]
    
    # for _ in range(num_prediction):
    #     x = prediction_list[-look_back:]
    #     x = x.reshape((-1, n_seq, n_steps, n_features))
    #     # x = x.reshape((-1, 1, look_back, 1))
    #     out = model.predict(x)[0][0]
    #     prediction_list = np.append(prediction_list, out)
    # prediction_list = prediction_list[look_back-1:]
    x_input = close_data[-1].reshape((-1))
    print('firts input{}'.format(x_input))
    temp_input=list(x_input)
    lst_output=[]
    i=0
    while(i<=num_prediction):        
        if(len(temp_input)> look_back):
            x_input=array(temp_input[1:])            
            #print(x_input)
            x_input = x_input.reshape((1, n_steps, n_features))
            # print("{} day input {}".format(i,x_input))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)         
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.append(yhat[0][0])
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)         
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i=i+1
    

    # print(lst_output)
        
    return lst_output
    
def predict_dates(last_date, num_prediction):   
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

def save_plot(directory):
    datatime = datetime.datetime.today()
    datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)
    filename = directory + '/' + currency + '-' + datestr + '.png';
        
    if os.path.exists(filename): os.remove(filename)
        
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time
  
  
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

  
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
  
  
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
  

def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
  

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

num_epochs = 20
interval_between_predicted = 10
num_prediction = 200
# we have univariate variable that is why n_features = 1
intervals = num_prediction
n_features = 1
n_seq = 1
#time window
n_steps = 60
look_back = n_steps
look_back_focast = n_steps
current_abs_path = str(pathlib.Path().resolve())

datatime = datetime.datetime.today()
datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)
save_plot_dir = 'predictions/prediction-conv-dens-normalized/' + str(datestr)


data = pd.read_json('../../cryptocurrency_rates_history.json')
currency_pairs = data[['full_date', 'pair', 'ask']]
currencies = currency_pairs['pair'].drop_duplicates()
scaler = MinMaxScaler()

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
        

for currency in currencies:
    if currency == 'ETH-EUR':
        currency_pairs_eth = currency_pairs['pair'] == currency
        data_filtered = currency_pairs[currency_pairs_eth]
        currency_dates = pd.to_datetime(data_filtered['full_date'])
        currency_dates[:] = np.reshape(currency_dates, (-1))
        
        
        close_data = data_filtered['ask'].values.reshape((-1, 1))
        close_data_plot = close_data[:].reshape((-1))
        
        #normalize data with minMaxScaler
        close_data = scaler.fit_transform(close_data)
        # reshape again in 1D
        close_data = close_data.reshape((-1))        
    
        
        #currency to be analysed and trained
        saved_model_path = 'saved_models/lstm-covn-dens-normalized/' + currency       
        saved_model_dir = current_abs_path + '/' + saved_model_path   
        print(saved_model_path)
        print(saved_model_dir)
        

        window_size = 30
        train_set = seq2seq_window_dataset(close_data, window_size, batch_size=128)
    
        #build RNN
        if os.path.isdir(saved_model_dir):
            print('model was already saved')
            model = load_model(saved_model_dir)
        else:      
            print('model was not saved')
            model = keras.models.Sequential([          
            keras.layers.LSTM(32, activation = 'relu', return_sequences=True),
            keras.layers.LSTM(32, activation='relu', return_sequences=True),
            keras.layers.Dense(1),
            keras.layers.Lambda(lambda x: x * 200)])
          
          
        lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
        optimizer = keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
        model.compile(loss=keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        
        #train the model
        history = model.fit(train_set, epochs=1, callbacks=[lr_schedule], shuffle = False)
        
        #save the trained model
        model.save(saved_model_path)           

        prediction = model.predict(train_set)      
        
        # prediction = model_forecast(model, close_data.reshape((-1, 1)), window_size)
        # prediction = prediction[0:len(close_data), -1, 0]
        
        prediction = scaler.inverse_transform(prediction.reshape((-1,1)))    
        #reshape in 1D     
        prediction = prediction.reshape((-1))
        prediction_dates = currency_dates[0: len(prediction)]
        
        #reshape in 2D
        close_data = np.reshape(close_data, (-1, 1))
        #renormalize with minMaxScaler
        close_data = scaler.inverse_transform(close_data)
        #reshape in 1D
        close_data = np.reshape(close_data, (-1))
        
        # #reshape in 2D
        # forecast = np.reshape(forecast, (-1, 1))
        # #renormalize with minMaxScaler
        # forecast = scaler.inverse_transform(forecast)
        # #reshape in 1D
        # forecast = np.reshape(forecast, (-1))

        today = datetime.datetime.today()
        # forecast_startdate = today - relativedelta(minutes=look_back_focast*30)
        forecast_startdate = today

        forecast_dates = np.array([forecast_startdate + datetime.timedelta(minutes=30*x) for x in range(0, num_prediction + 1)])
            
        #plot the data
        plt.figure(figsize=(45, 25))
        ax = plt.gca()
        
        annotate_points(prediction_dates, prediction, ax, intervals)
        # annotate_points(forecast_dates, forecast, ax, interval_between_predicted)    
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
        # ax.set_yticks(np.arange(np.min(forecast) - 1, np.max(forecast) + 1, intervals))   
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        plt.plot(currency_dates[-500:], close_data_plot[-500:])
        plt.plot(prediction_dates[-500:], prediction[-500:])
        # plt.plot(forecast_dates, forecast, marker = 'o')
        
        additional_info = "Currency: " + currency + "\n" + '\n epoch: ' + str(num_epochs) + '\n window size: ' + str(n_steps)
            
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.figtext(.9, 0.9, additional_info)
        plt.legend(['original', 'trained', 'prediction'])
        ax.grid(True)
        save_plot(save_plot_dir)
    
    # plt.show()

















