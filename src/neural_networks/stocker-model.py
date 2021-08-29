# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:26:09 2021

@author: jeti8
"""


from stocker import Stocker

amazon = Stocker('NKE')

model, model_data = amazon.create_prophet_model(days=90)


amazon.evaluate_prediction(nshares=1000)

amazon.predict_future(days=10)
amazon.predict_future(days=100)