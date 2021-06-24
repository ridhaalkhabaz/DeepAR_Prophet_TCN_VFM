import sys
import csv
import random 
import math
import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf 
from preprocessing import multivariate_data, univariate_data

# to create a data set for our training and 
def configure(data, params = [0, 4557, 5535, 7933], past_history=300, forecast_horizon=5, batch_size=64,  multi = False, column_index=7):
    if multi:
        x_mult, y_mult= multivariate_data(np.array(data), params[0], params[1], \
            past_history, forecast_horizon, column_index=column_index)
        valX_mult , valY_mult = multivariate_data(np.array(data), params[1], \
            params[2], past_history, forecast_horizon, column_index=column_index)
        testX_mult, testY_mult = multivariate_data(np.array(data), params[2], params[3],\
             past_history, forecast_horizon,  column_index=column_index)
        # to avoid combatibility issues
        x_mult = x_mult.astype(np.float32)
        y_mult = y_mult.astype(np.float32)
        valX_mult = valX_mult.astype(np.float32)
        valY_mult = valY_mult.astype(np.float32)
        testX_mult = testX_mult.astype(np.float32)
        testY_mult = testY_mult.astype(np.float32)
        # to build data set
        train_data_mult = tf.data.Dataset.from_tensor_slices((x_mult, y_mult)).cache().batch(batch_size).repeat()
        val_data_mult = tf.data.Dataset.from_tensor_slices((valX_mult, valY_mult)).batch(batch_size).repeat()
        return train_data_mult, val_data_mult, testX_mult, testY_mult
    # for univariate 
    x_uni, y_uni = univariate_data(np.array(data), params[0], params[1], past_history, forecast_horizon)
    valX_uni , valY_uni = univariate_data(np.array(data), params[1], params[2], past_history, forecast_horizon)
    testX_uni, testY_uni = univariate_data(np.array(data),params[2], params[3], past_history, forecast_horizon)
    # to avoid combatibality issues
    x_uni = x_uni.astype(np.float32)
    y_uni = y_uni.astype(np.float32)
    valX_uni = valX_uni.astype(np.float32)
    valY_uni = valY_uni.astype(np.float32)
    testX_uni = testX_uni.astype(np.float32)
    testY_uni = testY_uni.astype(np.float32)
    # to build data set 
    train_data_uni = tf.data.Dataset.from_tensor_slices((x_uni, y_uni)).cache().batch(batch_size).repeat()
    val_data_uni = tf.data.Dataset.from_tensor_slices((valX_uni, valY_uni)).batch(batch_size).repeat()
    return train_data_uni, val_data_uni, testX_uni, testY_uni