# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 07:03:38 2021

@author: rodion kovalenko
"""

import tensorflow
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
from random import random
import datetime

print(tf.executing_eagerly())


data = pd.read_json('../../cryptocurrency_rates_history.json')

currency_pairs = data[['full_date', 'pair', 'ask']]

currencies = currency_pairs['pair'].drop_duplicates()
date_list = [datetime.datetime.today() + datetime.timedelta(minutes=30*x) for x in range(0, 5001)]

for currency in currencies:
    if currency == 'ETH-EUR':
        currency_pairs = data[['full_date', 'pair', 'ask']]
        currency_pairs_eth = currency_pairs['pair'] == currency
    
        start_date = "2021-03-23"
        end_date = "2021-08-10"
    
        after_start_date = currency_pairs["full_date"] >= start_date
        before_end_date = currency_pairs["full_date"] <= end_date
        between_two_dates = after_start_date & before_end_date
    
        currency_pairs = currency_pairs[currency_pairs_eth]
    
        currency_pairs = currency_pairs.loc[between_two_dates]
    
        currency_pairs['full_date'] = pd.to_datetime(currency_pairs['full_date'])
    
        print(currency_pairs.plot(kind="line", x='full_date', y='ask', style='-o', xlabel = currency))
        
 
        # fit model
        model = ARIMA(currency_pairs['ask'].to_numpy(), order=(1, 1, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(currency_pairs['ask'].to_numpy()), len(currency_pairs['ask'].to_numpy()) + 5000, typ='levels')        
        
        dates = matplotlib.dates.date2num(date_list)
        matplotlib.pyplot.plot_date(dates, yhat)
        
        print(yhat)

       # print(currency_pairs)











