import sys
import csv
import random 
from datetime import datetime
import math
import numpy as np
from numpy import array
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import datetime as dttm


# Convert the sequence of seconds into datetime array
def parser(x):
    """
    This is a parser for seconds. parameters are: 
    x : ARRAY-LIKE: the seconds columns. 
    """
    now_seconds = 0
    y = x.astype(np.float) + now_seconds
    z = pd.to_datetime(y, unit='s')
    return z

# Read the data from fname and eventually plot them
def read_data_slugging(fname,params=[10, 0, 5e-3, 10],is_not_clean=False):
    """
    we are normalizing and add some removed values, gaussian noise, and spikes. parameters are:
    - fname: STRING: the name of the file or a path to it
    - params: the parameters used for spikes, gaussian noise, and removed values, it goes like the following:
        - params[0]: is the number of spikes desired to be added to our data set.
        - params[1]: is the mean for the gaussian noise 
        - params[2]: is the standard deviation for the gaussian noise 
        - params[3]: is the number of removed values desired to be added to our data set.
    - is_not_clean: BOOLEAN: whether you want to clean the data set or not. 
    """
    # Read the time series
    datats = read_csv(fname, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

    headers = list(datats.columns.values)
    headers.insert(0, datats.index.name)

    # Resample the data using a uniform timestep
    datats = datats.resample('S').mean()
    datats = datats.interpolate(method='linear')

    # Convert data to numpy array
    data = datats.reset_index().values

    # Replace timestamps with seconds
    time_sec = array([data[i, 0].timestamp() for i in range(len(data))])
    data = np.c_[time_sec, data[:, 1:]]

    if is_not_clean:
        # Adding spikes:
        data = spikes_addition(data, num_points=int(params[0]), is_not_well=True)


    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    scaler.scale_[0] = 1    # Do not normalize time

    # Apply the same normalization to all pressure readings
    pind = list(range(1, 8))  # Indices of pressure readings
    pmin = scaler.data_min_[pind].min()
    pmax = scaler.data_max_[pind].max()
    scaler.scale_[pind] = ((scaler.feature_range[1] - scaler.feature_range[0]) / (pmax - pmin))
    scaler.min_[pind] = scaler.feature_range[0] - pmin * scaler.scale_[pind]

    data = scaler.transform(data)
    if is_not_clean:
        # Adding gaussian noise 
        data = gaussian_noise_addition(data, mu=params[1], sigma=params[2])
         # Removing some points:
        data, _ = random_missing_data(data, num_points=int(params[3]))
    return data, scaler

# This function is to add spikes to the data before normalization
def spikes_addition(data, num_points=0, is_not_well=False):
    """
    Here, we add spikes to the orginal data per column. the spikes range is within
    the maximum and minimum of each column. The parameters are:
    - num_points: LIST: the number of observations to be missed.
    - is_well: BOOLEAN: is the data a slugging data set or a well data set
    """
    for i in random.sample(range(0, len(data)), num_points):
        # we begin at the second column because we assume that the first column is time 
        data[i][1] = np.random.uniform(data[:,1].min(),data[:,1].max())
        data[i][2] = np.random.uniform(data[:,2].min(),data[:,2].max())
        data[i][3] = np.random.uniform(data[:,3].min(),data[:,3].max())
        data[i][4] = np.random.uniform(data[:,4].min(),data[:,4].max())
        data[i][5] = np.random.uniform(data[:,5].min(),data[:,5].max())
        if is_not_well :
            data[i][6] = np.random.uniform(data[:,6].min(),data[:,6].max())
            data[i][7] = np.random.uniform(data[:,7].min(),data[:,7].max())
            data[i][8] = np.random.uniform(data[:,8].min(),data[:,8].max())
            data[i][9] = np.random.uniform(data[:,9].min(),data[:,9].max())
    return data

# This function is to add gausssian noise to our data set after normalization
def gaussian_noise_addition(data, mu=0, sigma=0.005):
    """
    we are simply adding gaussain noise to our data set. parameters are:
    - mu: FLOAT: the mean. 
    - sigma: FLOAT: the standard deviation. 
    """
    # we begin at the second column because we assume that the first column is time 
    noise = np.random.normal(mu, sigma, data[:, 1:].shape)
    data[:, 1:] = data[:, 1:] + noise
    return data

# This function is to remove some values of the data set. 
def random_missing_data(data, num_points=0):
    """
    we are removing some observations. parameters are:
    -num-points: INTEGER: the number of observations you want to remove from the data set. 
    """
    from random import randint
    index = [randint(0, len(data)) for i in range(num_points)]
    # we begin at the second column because we assume that the first column is time
    data[index, 1:10] = 0
    return data, index

# This function is to generate train samples for univariate TCN models 
def univariate_data(tseries, start_index, end_index, history_size, forecast_horizon):
    """
    This is to make a suitable data set for ML algoritms. parameters are:
    - tseries: NP.ARRAY: the data set
    - start_index:INTEGER: the starting point is at
    - end_index:INTEGER: the ending point is at 
    - hisotry_size_:INTEGER: the number of past observations to use in our training for a single prediction. 
    - forecast_horizon:INTEGER: the number of future points you want to predict. 
    """

    data = []
    labels = []

    start_index = start_index + history_size

    for i in range(start_index, end_index - forecast_horizon + 1, forecast_horizon):
        indices = range(i - history_size, i, 1)
        data.append(np.reshape(tseries[indices], (history_size, 1)))
        labels.append(tseries[i:i + forecast_horizon])

    return np.array(data), np.array(labels)

# This function is to generate train samples for multivariate TCN models 
def multivariate_data(tseries, start_index, end_index, history_size, forecast_horizon, column_index=7):
    """
    This is to make a suitable data set for ML algoritms. parameters are:
    - tseries: NP.ARRAY: the data set
    - start_index:INTEGER: the starting point is at
    - end_index:INTEGER: the ending point is at 
    - hisotry_size_:INTEGER: the number of past observations to use in our training for a single prediction. 
    - forecast_horizon:INTEGER: the number of future points you want to predict. 
    - column_index: INTEGER: the index of the targeted column 
    """
    data = []
    labels = []
    for i in range(start_index, end_index -history_size +1, forecast_horizon):
        end = i+history_size
        # if len(tseries[end:end+forecast_horizon, wanted:])
        data.append(tseries[i:end, :])
        labels.append(tseries[end:end+forecast_horizon, column_index:])
    return np.array(data), np.array(labels)

def read_data_well(fname, params=[0, 0, 5e-3, 100], is_not_clean=False):
    """
    we are normalizing and add some removed values, gaussian noise, and spikes. parameters are:
    - fname: STRING: the name of the file or a path to it
    - params: LIST: the parameters used for spikes, gaussian noise, and removed values, it goes like the following:
        - params[0]: is the number of spikes desired to be added to our data set.
        - params[1]: is the mean for the gaussian noise 
        - params[2]: is the standard deviation for the gaussian noise 
        - params[3]: is the number of removed values desired to be added to our data set.
    - is_not_clean: BOOLEAN: whether you want to clean the data set or not. 
    """
    # Read the time series
    datats = read_csv(fname, header=0, dayfirst=True, parse_dates=[0], index_col=0, squeeze=True)  # , date_parser=parser

    headers = list(datats.columns.values)
    headers.insert(0, datats.index.name)

    # Convert data to numpy array
    data = datats.reset_index().values

    # Split data into flow periods, and resample each flow period using a uniform timestep
    dt = np.ediff1d(data[:, 0])
    fpbreak = dttm.timedelta(hours=1)  # Minimal break between flow periods
    dt = dt - fpbreak
    ind = np.where(dt - fpbreak > pd.Timedelta(0))[0]
    ind = np.r_[ind, len(data)-1]

    Nfp = len(ind)  # Number of flow periods
    fp = ['None'] * Nfp
    n0 = 0
    n1 = ind[0]+1
    for n in range(Nfp):
        # Resample each flow period separately
        fpts = datats[n0:n1].resample('T').mean()
        fpts = fpts.interpolate(method='linear')
        # Save the resampled flow period to a list of numpy arrays
        fp[n] = fpts.reset_index().values
        #fp[n] = data[n0:n1,:]
        n0 = n1
        if n+1 < Nfp:
            n1 = ind[n+1] + 1   
 
    # Get the normalization parameters for all data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data[:,1:]) # Exclude Datetime from normalization

    # Normalize every flow period
    for n in range(Nfp):
        if is_not_clean:
            fp[n] = spikes_addition(fp[n], num_points=params[0], is_not_well=False)
        fp[n][:,1:] = scaler.transform(fp[n][:,1:])
        if is_not_clean:
            #adding some gaussian noise 
            fp[n] = gaussian_noise_addition(fp[n], mu=params[1], sigma=params[2])
            # Removing some points:
            fp[n], _ = random_missing_data(fp[n], num_points=int(params[3]))

    return fp, headers, scaler
