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
from matplotlib import pyplot, rcParams, dates
import matplotlib.pyplot as plt


# Convert the sequence of seconds into datetime array
def parser(x):
    """
    This is a parser for seconds. The parameters are: 
    - x : ARRAY-LIKE: the seconds columns. 
    IT returns:
    - z: ARRAY-LIKE: the array of seconds 
    """
    now_seconds = 0
    y = x.astype(np.float) + now_seconds
    z = pd.to_datetime(y, unit='s')
    return z

# Read the data from fname and eventually plot them
def read_data_slugging(fname,params=[10, 0, 5e-3, 10],is_not_clean=False, plot_data=False):
    """
    we are normalizing and add some removed values, gaussian noise, and spikes. The parameters are:
    - fname: STRING: the name of the file or a path to it
    - params: the parameters used for spikes, gaussian noise, and removed values, it goes like the following:
        - params[0]: Integer: is the number of spikes desired to be added to our data set.
        - params[1]: Integer: is the mean for the gaussian noise 
        - params[2]: Integer: is the standard deviation for the gaussian noise 
        - params[3]: Integer: is the number of removed values desired to be added to our data set.
    - is_not_clean: BOOLEAN: whether you want to clean the data set or not. 
    - plot_data: BOOLEAN: whether you want to plot the original data
    IT returns:
    - df: DATAFRAME: the organized pandas data frame of the original data . 
    - data: ARRAY-LIKE: the original data. 
    - headers: LIST: the list of column names of the original data set. 
    - scaler: SCALER: the scaler used for normalization
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
    
    # this is to test robustness in a vague sense. 
    if is_not_clean:
        # Adding spikes:
        data = spikes_addition(data, num_points=int(params[0]), is_not_well=True)
    # to plot the data 
    if (plot_data):
        color = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        fig0, ax = pyplot.subplots()
        # pressure and temperature readings 
        ax.plot(data[:, 0], data[:, 1:8])
        ax.set_xlabel(headers[0])
        ax.set_ylabel('Pressure (bar)')
        headersplot = [w[-8:-2] for w in headers[1:8]]
        px = ['$p(x_{%d}' % i for i in range(1, 8)]
        tail = [')$'] * 7
        headersplot = [px + headersplot + tail for px, headersplot, tail in zip(px, headersplot, tail)]
        ax.legend(headersplot)
        ax.set_title('Distributed pressure readings')
        fig0.savefig('pressure-readings-slugging.jpeg')
        # mass rate measurements 
        fig, ax1 = pyplot.subplots()
        
        ax2 = ax1.twinx()
        ln1 = ax1.plot(data[:, 0], data[:, 8], color=color[1], label='Gass mass rate (kg/sec)')
        ln2 = ax2.plot(data[:, 0], data[:, 9], color=color[2], label='Liquid mass rate (kg/sec)')
        pyplot.subplots_adjust(right=0.85)
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Gas Mass rate (kg/sec)', color=color[1])
        ax1.tick_params(axis='y', colors=color[1])
        ax2.set_ylabel('Liquid Mass rate (kg/sec)', color=color[2])
        ax2.tick_params(axis='y', colors=color[2])
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')
        ax1.set_title('Mass Flowrates (kg/sec)')
        fig.savefig('mass-flowrates-slugging.jpeg')

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
    # this is to test robustness. 
    if is_not_clean:
        # Adding gaussian noise 
        data = gaussian_noise_addition(data, mu=params[1], sigma=params[2])
         # Removing some points:
        data, _ = random_missing_data(data, num_points=int(params[3]))
    df = pd.DataFrame.from_dict(dict(zip(['ds', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7','gas(kg/sec)', 'liquid(kg/sec)' ], data.T)))
    return df, data, headers, scaler


# This function is to add spikes to the data before normalization
def spikes_addition(data, num_points=0, is_not_well=False):
    """
    Here, we add spikes to the orginal data per column. the spikes range is within
    the maximum and minimum of each column. Here, we hard code the columns for convenience purposes.  the  The parameters are:
    - data: ARRAY-LIKE: the original data set. 
    - num_points: LIST: the number of observations to be missed.
    - is_well: BOOLEAN: is the data a slugging data set or a well data set
    IT returns:
    - data: ARRAY-LIKE: the modified data. 
    """
    ## TODO try not hard coding by using .shape
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
    - data: ARRAY-LIKE: the original data set. 
    - mu: FLOAT: the mean. 
    - sigma: FLOAT: the standard deviation. 
    IT returns: 
    - data: ARRAY-LIKE: the modified data. 
    """
    # we begin at the second column because we assume that the first column is time 
    noise = np.random.normal(mu, sigma, data[:, 1:].shape)
    data[:, 1:] = data[:, 1:] + noise
    return data

# This function is to remove some values of the data set. 
def random_missing_data(data, num_points=0):
    """
    we are removing some observations. parameters are:
    - data: ARRAY-LIKE: the original data set. 
    - num_points: INTEGER: the number of observations you want to remove from the data set. 
    IT returns: 
    - data: ARRAY-LIKE: the modified data. 
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
    - forecast_horizon: INTEGER: the number of future points you want to predict. 
    IT returns: 
    - data: ARRAY-LIKE: the desired input for univariate model, i.e. past readings or pressure or temperature. 
    - labels: ARRAY-LIKE: the target for univariate model, i.e. gas/water/oil flow rate. 
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
    - hisotry_size :INTEGER: the number of past observations to use in our training for a single prediction. 
    - forecast_horizon:INTEGER: the number of future points you want to predict. 
    - column_index: INTEGER: the index of the targeted column 
    IT returns:
    - data: ARRAY-LIKE: the desired input for multivariate model, i.e. past readings or pressure or temperature or time. 
    - labels: ARRAY-LIKE: the target for multivariate model, i.e. gas/water/oil flow rate. 
    """
    data = []
    labels = []
    for i in range(start_index, end_index -history_size +1, forecast_horizon):
        end = i+history_size
        # here we assume that the targeted column has the input before it. Hence that is why we have it like the following. 
        data.append(tseries[i:end, :column_index])
        labels.append(tseries[end:end+forecast_horizon, column_index:])
    return np.array(data), np.array(labels)

def read_data_well(fname, params=[0, 0, 5e-3, 100], is_not_clean=False, plot_data=False):
    """
    we are normalizing and add some removed values, gaussian noise, and spikes. parameters are:
    - fname: STRING: the name of the file or a path to it
    - params: LIST: the parameters used for spikes, gaussian noise, and removed values, it goes like the following:
        - params[0]: is the number of spikes desired to be added to our data set.
        - params[1]: is the mean for the gaussian noise 
        - params[2]: is the standard deviation for the gaussian noise 
        - params[3]: is the number of removed values desired to be added to our data set.
    - is_not_clean: BOOLEAN: whether you want to clean the data set or not. 
    - plot_data: BOOLEAN: whether you want to plot the original data set as two figures, one for the input and one for the output. 
    IT returns: 
    - data: ARRAY-LIKE: the original data set. 
    - headers: LIST: the names of the columns of the original data set. 
    - scaler: SCALER: the scaler used to normlaize the data. 
    """
    # Read the time series
    datats = read_csv(fname, header=0, dayfirst=True, parse_dates=[0], index_col=0, squeeze=True) 

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
    # Plot the graphs
    if (plot_data):

        color = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        dfmt = dates.DateFormatter('%b %d') # Month day

        # Pressure and temperature
        fig1, ax1 = pyplot.subplots()
        ax2 = ax1.twinx()
        for n in range(Nfp):
            if n == 0:
                hl1 = ax1.plot(fp[n][:, 0], fp[n][:, 1], color=color[1], label='Pressure')
                hl2 = ax2.plot(fp[n][:, 0], fp[n][:, 2], color=color[2], label='Temperature')
            else:
                ax1.plot(fp[n][:, 0], fp[n][:, 1], color=color[1])
                ax2.plot(fp[n][:, 0], fp[n][:, 2], color=color[2])

        ax1.xaxis.set_major_formatter(dfmt)
        fig1.autofmt_xdate()
        ax1.set_ylabel(headers[1], color=color[1])
        ax1.tick_params(axis='y', colors=color[1])
        headers[2] = headers[2].replace('degC', '°C')
        ax2.set_ylabel(headers[2], color=color[2])
        ax2.tick_params(axis='y', colors=color[2])

        hl = hl1 + hl2
        labs = [h.get_label() for h in hl]
        ax1.legend(hl, labs, loc=2)
        ax1.set_title('Pressure and temperature data')
        fig1.savefig('pressure-temperature-readings-welltest.jpeg')

        # Flow rates
        fig2, ax1 = pyplot.subplots()
        ax2 = ax1.twinx()
        for n in range(Nfp):
            if n == 0:
                hl1 = ax1.plot(fp[n][:, 0], fp[n][:, 3], color=color[3], label='Oil rate')
                hl2 = ax1.plot(fp[n][:, 0], fp[n][:, 4], color=color[4], label='Water rate')
                hl3 = ax2.plot(fp[n][:, 0], fp[n][:, 5], color=color[5], label='Gas rate')
            else:
                ax1.plot(fp[n][:, 0], fp[n][:, 3], color=color[3])
                ax1.plot(fp[n][:, 0], fp[n][:, 4], color=color[4])
                ax2.plot(fp[n][:, 0], fp[n][:, 5], color=color[5])

        ax1.xaxis.set_major_formatter(dfmt)
        fig2.autofmt_xdate()
        rheader = headers[3].split()[0] + ' & ' + headers[4]
        ax1.set_ylabel(rheader, color=color[3])
        ax1.tick_params(axis='y', colors=color[3])
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.set_ylabel(headers[5], color=color[5])
        ax2.tick_params(axis='y', colors=color[5])

        hl = hl1 + hl2 + hl3
        labs = [h.get_label() for h in hl]
        ax1.legend(hl, labs, loc=1)
        ax1.set_title('Flow rates data')
        fig2.savefig('mass-flowrate-welltest.jpeg')


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
# this function is used to get the difference between timestamps in seconds
def get_sec(time_str):
    """
    this function is get amount of seconds since the start of the month. the parameters are:
    - time_str: STRING: the timestamp
    IT returns:
    - Integer: the number of seconds since the start of the month. 
    """
    a, b = time_str.split(' ')
    h, m, s = b.split(':')
    k,c,n = a.split('-')
    # here we assume that every operation period is within the same month 
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(n)-1)*60*60*24
# this function is used to do some padding between operation periods. 
def dataframe(fp, col_names, padding, timing):
    """
    this function is to transform the well's data and do some padding if needed. The parameters are:
    - fp: LIST: the original data set, here is a list of array flow periods . 
    - col_names: LIST: the names of the columns. 
    - padding: BOOLEAN: whether you want to do some padding or not.
    - timing: BOOLEAN: whether you want to add some increasing series to count for the time series. 
    IT returns:
    - df: DATAFRAME: the organized dataframe of the original data set. 
    """
    df = pd.DataFrame.from_dict(dict(zip(col_names, fp[0].T)))
    for i in range(1, len(fp)):
        tmp = pd.DataFrame.from_dict(dict(zip(col_names, fp[i].T)))
        if padding:
            time_a = str(df.iloc[-1].ds)
            time_b = str(tmp.iloc[0].ds)
            minutes = int((get_sec(time_b)-get_sec(time_a))/60) - 1
            df_try = pd.DataFrame([df.iloc[-1]]*minutes)
            df = pd.concat([df, df_try], axis=0)
        df = pd.concat([df, tmp], axis=0)
    if timing:
        df['time'] = [i for i in range(len(df))]
    return df
# this function is to turn time into timestamp for our deepAR
def time_to_timestamp(seconds):
    """
    this function is to turn seconds into time stamps. The parameters are:
    - seconds: Integer: the number of seconds you have. 
    IT returns: 
    - timestamp: DATETIME TIMESTAMP: the time converted into a timestamp. 
    """
    addition = 1625893200
    minutes = seconds * 60 
    minutes += addition
    timestamp = dttm.datetime.fromtimestamp(minutes)
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')
# this function is to plot the results from the DeepAR algorithm 
def plot_prob_forecasts(ts_entry, forecast_entry):
    """
    This function is to plot the DeepAR results. The parameters are:
    - ts_entry: LIST: the original testing data set. 
    - forecast_entry: LIST: the forecasted observations. 
    """
    plot_length = 1000
    prediction_intervals = (80.0, 95.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()
