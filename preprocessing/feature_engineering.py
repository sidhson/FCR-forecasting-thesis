'''
Feature engineering functions for e.g. encoding time, categorical variables and normalization.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def encode_is_home(time_series):
    '''
    Encodes binary variable `is_home`. 
    
    Encode with map function is order to preserve NaN values. 
    Binary variables are casted as floats to ensure same datatype even if NaN values are non-existent. 
    '''
    time_series['is_home'] = time_series['is_home'].map({False:0, True:1}).astype(float)
    return time_series


def diff_odometer(time_series):
    '''
    Computes the difference between subsequent `odometer` measures.

    Returns an EV time series with the `odometer` attribute exchanged for `odometer_diff`.
    
    As odometer is strictly non-decreasing, the problem arises 
    with normalizing it for model training. Z-score or minmax scaling will
    inevetibly be flawed, as the scalers used will be rendered obsolete in 
    validation and test data sets, as odometer increases with time. 
    '''
    time_series['odometer_diff'] = time_series['odometer']-time_series['odometer'].shift(1)
    time_series = time_series.drop('odometer',axis = 1)
    return time_series


def create_dummies(time_series):
    '''
    Encodes `isPluggedIn`, `is_home` and `charging_status`. 
    Encodes with map function is order to preserve NaN values. 
    Binary variables are casted as floats to ensure same datatype even if NaN values are non-existent. 
    '''
    time_series['isPluggedIn'] = time_series['isPluggedIn'].map({False:0, True:1}).astype(float)
    time_series['is_home'] = time_series['is_home'].map({False:0, True:1}).astype(float)

    time_series['charge_status_CHARGING'] = time_series['charge_status'].map({'CHARGING':1, 'FULLY_CHARGED':0, 'NOT_CHARGING':0}).astype(float)
    time_series['charge_status_FULLY_CHARGED'] = time_series['charge_status'].map({'FULLY_CHARGED':1, 'CHARGING':0, 'NOT_CHARGING':0}).astype(float)
    time_series['charge_status_NOT_CHARGING'] = time_series['charge_status'].map({'NOT_CHARGING':1, 'CHARGING':0, 'FULLY_CHARGED':0}).astype(float)

    time_series = time_series.drop('charge_status', axis = 1)
    return time_series


def norm_coordinates(time_series,EV_constants,keep_original = False):
    '''
    DEPRICIATED.
    '''
    home_lat, home_long = list(map(float,EV_constants.home_location.values[0].rstrip(')').lstrip('(').split(',')))
    time_series['norm_lat'] = time_series['latitude']-home_lat
    time_series['norm_long'] = time_series['longitude']-home_long
    if not keep_original:
        time_series = time_series.drop('latitude',axis = 1)
        time_series = time_series.drop('longitude',axis = 1)
    return time_series


def encode_time(time_series):
    '''
    Encodes time-date attribute as pairs of sinus-consinus signals. 
    One pair for time of day, and one pair for time of week.

    Returns the EV time series with `time` exchanged for `day_sin`, `day_cos`, `week_sin`, `week_cos`.
    '''
    date_time = pd.to_datetime(time_series.pop('time'))
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    # ^ Encodes timestamps to seconds passed since some date. i.e. converts time to scalar value in unit seconds. 
    day = 24*60*60 # day in seconds
    week = 7*day # week in seconds
    time_series['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    time_series['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    time_series['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    time_series['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    return time_series


def normalize_3D_data(X_train, X_test, y_train, y_test, X_val=None, y_val=None, set_scaler='MinMaxScaler'):
    '''
    Normalizes 3D data for input X and output Y with Sklearn MinMaxScaler or StandardScaler. 
    If validation data is supplied, then it is scaled with the training data. 
    '''
    scalers = {'MinMaxScaler' : MinMaxScaler, 'StandardScaler' : StandardScaler}
    scaler = scalers[set_scaler]
    
    scalers_X = {}
    for i in range(X_train.shape[1]):
        scalers_X[i] = scaler()
        X_train[:, i, :] = scalers_X[i].fit_transform(X_train[:, i, :])
        X_test[:, i, :] = scalers_X[i].transform(X_test[:, i, :]) 

    ## Scales Y by "feature" (in this case the timestep in horizon). 
    ## Correct scaling independent of wether Y is (n,1) or (n,h).
    scaler_Y = scaler()
    y_train = scaler_Y.fit_transform(y_train)
    y_test = scaler_Y.transform(y_test)

    if X_val is not None and y_val is not None:
        for i in range(X_train.shape[1]):
            X_val[:, i, :] = scalers_X[i].transform(X_val[:, i, :])
        y_val = scaler_Y.transform(y_val)
        return X_train, X_test, y_train, y_test, X_val, y_val, scalers_X, scaler_Y
    else:
        return X_train, X_test, y_train, y_test, None, None, scalers_X, scaler_Y


def normalize_3D_data_notY(X_train, X_test, X_val=None, set_scaler='MinMaxScaler'):
    '''
    Normalizes 3D data for only input X with Sklearn MinMaxScaler or StandardScaler. 
    If validation data is supplied, then it is scaled with the training data. 
    '''
    scalers = {'MinMaxScaler' : MinMaxScaler, 'StandardScaler' : StandardScaler}
    scaler = scalers[set_scaler]
    
    scalers_X = {}
    for i in range(X_train.shape[1]):
        scalers_X[i] = scaler()
        X_train[:, i, :] = scalers_X[i].fit_transform(X_train[:, i, :])
        X_test[:, i, :] = scalers_X[i].transform(X_test[:, i, :]) 

    ## Scales Y by "feature" (in this case the timestep in horizon). 
    ## Correct scaling independent of wether Y is (n,1) or (n,h).
    if X_val is not None:
        for i in range(X_train.shape[1]):
            X_val[:, i, :] = scalers_X[i].transform(X_val[:, i, :])
        return X_train,X_val, X_test, scalers_X
    else:
        return X_train, None,X_test, scalers_X


def normalize_2D_data_notY(X_train, X_test, X_val=None, set_scaler='MinMaxScaler'):
    '''
    Normalizes 2D data for only input X with Sklearn MinMaxScaler or StandardScaler. 
    If validation data is supplied, then it is scaled with the training data. 
    '''
    
    scalers = {'MinMaxScaler' : MinMaxScaler, 'StandardScaler' : StandardScaler}
    scaler = scalers[set_scaler]()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if X_val is not None:
        X_val = scaler.transform(X_val)
        return X_train,X_val,X_test
    else:
        return X_train,X_test,None