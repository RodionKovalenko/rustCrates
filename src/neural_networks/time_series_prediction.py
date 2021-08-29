# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:47:17 2021

@author: jeti8
"""
import tensorflow
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from random import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow_addons.layers import MultiHeadAttention
import datetime
import matplotlib.dates as mdates
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

print(tf.executing_eagerly())
intervals = 50
date_list = [datetime.datetime.today() + datetime.timedelta(minutes=30*x) for x in range(0, intervals + 1)]


def predict_moving_average(data, predict_start=None, intervals=15):
    data_norm = np.reshape(scaler.fit_transform(data.reshape(-1, 1)), (-1))
    model = ARIMA(data_norm, order=(2, 1, 5))
    model_fit = model.fit()

    if predict_start is None:
        predict_start = len(data) - 2

    prediction_norm = model_fit.predict(predict_start, predict_start + intervals, typ='levels')
    return np.reshape(scaler.inverse_transform(prediction_norm.reshape(-1,1)), (-1))


def predict_autoregression(data, predict_start=None, intervals=15):
    model = AutoReg(data, lags=1, old_names=False)
    model_fit = model.fit()
    
    if predict_start is None:
        predict_start = len(data) - 2
        
    return model_fit.predict(predict_start, predict_start + intervals)


def predict_autoregressive_moving_average(data, predict_start=None, intervals=15):
    model = ARIMA(data, order=(2, 0, 1))
    model_fit = model.fit()

    if predict_start is None:
        predict_start = len(data) - 2

    return model_fit.predict(predict_start, predict_start + intervals)



def predict_autoregressive_integrated_moving_verage(data, predict_start=None, intervals=15):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()

    if predict_start is None:
        predict_start = len(data) - 2

    return model_fit.predict(predict_start, predict_start + intervals, typ='levels')

def predict_simple_exponential_smoothing(data, predict_start=None, intervals=15):
    model = SimpleExpSmoothing(data, initialization_method='estimated')
    model_fit = model.fit()

    if predict_start is None:
        predict_start = len(data) - 2

    return model_fit.predict(predict_start, predict_start + intervals)


def get_training_data(raw_data):
    return {"inputs":  {16, len(raw_data['ask'].to_numpy()), raw_data['ask'].to_numpy(), raw_data['full_date'].to_numpy()}}

data = pd.read_json('../../cryptocurrency_rates_history.json')
currency_pairs = data[['full_date', 'pair', 'ask']]
currencies = currency_pairs['pair'].drop_duplicates()


for currency in currencies:
    currency_pairs = data[['full_date', 'pair', 'ask']]
    currency_pairs_eth = currency_pairs['pair'] == currency

    start_date = "2021-03-23"
    end_date = "2021-08-24"

    after_start_date = currency_pairs["full_date"] >= start_date
    before_end_date = currency_pairs["full_date"] <= end_date
    between_two_dates = after_start_date & before_end_date

    currency_pairs = currency_pairs[currency_pairs_eth]
    currency_pairs = currency_pairs.loc[between_two_dates]
    currency_pairs['full_date'] = pd.to_datetime(currency_pairs['full_date'])

    # if currency == 'ETH-EUR':
    # fit model
    print('moving average ' + currency)
    alg = 'moving_average'
    yhat = predict_moving_average(currency_pairs['ask'].to_numpy(), intervals=intervals)
    print(yhat)

    # print('autoregression')    
    # alg = 'autoregression'       
    # yhat = predict_autoregression(currency_pairs['ask'].to_numpy(), intervals=intervals)
    # print(yhat)

    # print('autoregressive moving average')      
    # alg = 'arma'
    # yhat = predict_autoregressive_moving_average(currency_pairs['ask'].to_numpy(), intervals=intervals)
    # print(yhat)

    # print('Autoregressive Integrated Moving Average')      
    # alg = 'arima'
    # yhat = predict_autoregressive_integrated_moving_verage(currency_pairs['ask'].to_numpy(), intervals=intervals)
    # print(yhat)
    
    # print('Simple Exponential Smoothing')       
    # alg = 'simple_exponential_smoothing'
    # yhat = predict_simple_exponential_smoothing(currency_pairs['ask'].to_numpy(), intervals=intervals)
    # print(yhat)
    
    # model = ModelTrunk(num_layers = 3)
    # training_data = get_training_data(currency_pairs)
    # model.call()

    fig, ax = plt.subplots(1,1)   
 
    xs = date_list
    ys = yhat      
    
    ax.set_title('Prediction ma ' + currency)
    plt.rcParams["figure.figsize"] = (25, 10)
    ax.plot(xs, ys, 'bo-')
    
    interval = (np.max(yhat) - np.min(yhat) + 0.1) 
    
    print(interval)
    
    for i, (x, y) in enumerate(zip(xs, ys)):   

        if (i % 3 == 0):
            label ="{:.2f}".format(y)
        
            ax.annotate(label, # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0, 2), # distance from text to points (x,y)
                         ha='left') # 
    
    ax.set_yticks(np.arange(np.min(yhat) - 1, np.max(yhat) + 1, interval))        
    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction price')
    ax.legend([currency])
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=120))   #to get a tick every 15 minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))     #optional formatting 
    
    datatime = datetime.datetime.today()
    datestr = '-' + str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour)
    filename = 'predictions/prediction' + datestr + '/' + currency + alg + '.png';
    
    if os.path.exists(filename): os.remove(filename)
            
    Path("predictions/prediction" + datestr).mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)
    




class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x


class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

        
    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out












