import sys
import csv
import random 
import math
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, RepeatVector
from tensorflow.keras.preprocessing import sequence
from tcn import TCN
import tensorflow as tf


def model_init(model='Prophet', params=['ds','p1','p2','p3', 'p4', 'p5', 'p6', 'p7'], hyperparamters=[{'daily_seasonality':False,'weekly_seasonality':False, 'yearly_seasonality':False,'changepoint_prior_scale':12.0, 'seasonality_prior_scale':0.1, \
                        'holidays_prior_scale':0.01, 'seasonality_mode':'multiplicative'}]):
    if model=='Prophet':
            model = Prophet(**hyperparamters[0])
            for i in params:
                model.add_regressor(i)
    if model=='TCN':
            model = Sequential(
                layers=[
                    TCN(**hyperparamters[0]),  
                    Dense(hyperparamters[1]) 
                    ]
            )
            model.summary()
            model.compile('adam', 'mse')
    return model            


def model_train(data, target, model, model_kind, con, val):
    if model_kind=='Prophet':
        data = data.rename(columns={target:'y'})
        model.fit(data)
        return model  
    return model.fit(data, epochs=con[0], steps_per_epoch=con[1],validation_data=val, validation_steps=con[1])


def model_predict(data_test, model_kind, model, target):
    if model_kind=='Prophet':
        data_test = data_test.rename(columns={target:'y'})
        test_data = model.predict(data_test.drop(columns="y"))

        return test_data, mean_squared_error(data_test['y'], test_data['yhat'])
    return model.predict(data_test), mean_squared_error(data_test[target], model.predict(data_test).flatten())
