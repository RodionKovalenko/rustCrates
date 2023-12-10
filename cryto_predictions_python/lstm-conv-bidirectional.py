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

num_epochs = 20
interval_between_predicted = 10
num_prediction = 200
# we have univariate variable that is why n_features = 1
intervals = num_prediction
n_features = 1
n_seq = 1
#time window
n_steps = 64
look_back = n_steps
look_back_focast = n_steps
current_abs_path = str(pathlib.Path().resolve())

datatime = datetime.datetime.today()
datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)
save_plot_dir = 'predictions/prediction-raw/' + str(datestr)


data = pd.read_json('../../cryptocurrency_rates_history.json')
currency_pairs = data[['full_date', 'pair', 'ask']]
currencies = currency_pairs['pair'].drop_duplicates()
scaler = MinMaxScaler()


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
        
        close_data, close_data_y = split_sequence(close_data, n_steps)
        close_data = np.reshape(close_data, (close_data.shape[0], close_data.shape[1], n_features))    
        
        #currency to be analysed and trained
        saved_model_path = 'saved_models/lstm-trained-raw/' + currency       
        saved_model_dir = current_abs_path + '/' + saved_model_path   
        print(saved_model_path)
        print(saved_model_dir)

        
        #build RNN
        if os.path.isdir(saved_model_dir):
            print('model was already saved')
            model = load_model(saved_model_dir)
        else:      
            print('model was not saved')
            model = Sequential()            
            model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
            model.add(LSTM(50, activation='tanh'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
        
        #train the model
        model.fit(close_data, close_data_y, epochs=num_epochs, verbose=1, shuffle = False)
        
        #save the trained model
        model.save(saved_model_path) 
        
            
        prediction = model.predict(close_data)
          #renormalize with minMaxScaler
        prediction = scaler.inverse_transform(prediction)    
        #reshape in 1D     
        prediction = prediction.reshape((-1))
        prediction_dates = currency_dates[n_steps: len(prediction)]
        
      
        forecast = predict(num_prediction, model, n_seq, n_steps, n_features, look_back_focast, close_data)  
        
        
        #reshape in 2D
        close_data = np.reshape(close_data, (-1, 1))
        #renormalize with minMaxScaler
        close_data = scaler.inverse_transform(close_data)
        #reshape in 1D
        close_data = np.reshape(close_data, (-1))
        
        #reshape in 2D
        forecast = np.reshape(forecast, (-1, 1))
        #renormalize with minMaxScaler
        forecast = scaler.inverse_transform(forecast)
        #reshape in 1D
        forecast = np.reshape(forecast, (-1))

        today = datetime.datetime.today()
        # forecast_startdate = today - relativedelta(minutes=look_back_focast*30)
        forecast_startdate = today

        forecast_dates = np.array([forecast_startdate + datetime.timedelta(minutes=30*x) for x in range(0, num_prediction + 1)])
            
        #plot the data
        plt.figure(figsize=(45, 25))
        ax = plt.gca()
        
        annotate_points(prediction_dates, prediction, ax, intervals)
        annotate_points(forecast_dates, forecast, ax, interval_between_predicted)    
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
        ax.set_yticks(np.arange(np.min(forecast) - 1, np.max(forecast) + 1, intervals))   
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        plt.plot(currency_dates[-500:], close_data_plot[-500:])
        plt.plot(prediction_dates[-500:], prediction[-500:])
        plt.plot(forecast_dates, forecast, marker = 'o')
        
        additional_info = "Currency: " + currency + "\n" + '\n epoch: ' + str(num_epochs) + '\n window size: ' + str(n_steps)
            
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.figtext(.9, 0.9, additional_info)
        plt.legend(['original', 'trained', 'prediction'])
        ax.grid(True)
        save_plot(save_plot_dir)
    
    # plt.show()

















