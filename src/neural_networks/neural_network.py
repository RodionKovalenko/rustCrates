# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 07:03:38 2021

@author: rodion kovalenko
"""

import pandas as pd

data = pd.read_json('../../cryptocurrency_rates_history.json')

currency_pairs = data[['full_date', 'pair', 'ask']]

currencies = currency_pairs['pair'].drop_duplicates()


for currency in currencies:
    currency_pairs = data[['full_date', 'pair', 'ask']]
    currency_pairs_eth = currency_pairs['pair'] == currency
    
    start_date = "2021-03-01"
    end_date = "2021-06-24"
    
    after_start_date = currency_pairs["full_date"] >= start_date
    before_end_date = currency_pairs["full_date"] <= end_date
    between_two_dates = after_start_date & before_end_date

    
    currency_pairs = currency_pairs[currency_pairs_eth]
    
    currency_pairs = currency_pairs.loc[between_two_dates]
    
    currency_pairs['full_date'] = pd.to_datetime(currency_pairs['full_date'])
    
    
    currency_pairs.plot(kind="line", x='full_date', y='ask', style='-o', xlabel = currency)
