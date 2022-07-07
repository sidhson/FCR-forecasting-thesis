'''
Generates cross-sectional data for XGBoost models. 
Saves training, validation and testing datasets with splits (0.6, 0.2, 0.2). 
The sliding window used to generate the data takes the past 24 hours and the forecasted day one week prior as input. 

NOTE: DO NOT FORGET to set correct DATE and FCR before the script is run! 
'''

import numpy as np 
import os
import pandas as pd
import sys

DATE = '2022-05-16'
FCR = '2-D'

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)
datadir = f'{root}/data/processed_data/data_{DATE}'

from preprocessing.feature_engineering import *

if FCR == '1-D':
    HORIZON = int(48)
    DARK_PATCH = int(14) # Forecast created at 17.00
    LOOK_BACK = int(48)
elif FCR == '2-D':
    HORIZON = int(48)
    DARK_PATCH = int(20 + 48) # Forecast created at 14.00
    LOOK_BACK = int(48)

# Data in X is backwards, i.e. most recent is furthest to the left. 
prev_day = [i for i in range(1,49)]
prev_week = [i for i in range(7*24*2 - DARK_PATCH-HORIZON,7*24*2-DARK_PATCH+1)]
alt_lags = prev_day
alt_lags.extend(prev_week)

savedir = f'{root}/data/train_{FCR}/{DATE}/cross_data'
if not os.path.exists(savedir):
    os.mkdir(savedir)

FEATURES = ['day_sin', 'day_cos', 'week_sin', 'week_cos', 'odometer_diff', 'range', 'latitude', 'longitude', 'distance', 'isPluggedIn', 'charge_status_CHARGING', 'charge_status_FULLY_CHARGED', 'charge_status_NOT_CHARGING', 'soc', 'is_home']

dont_lag = ['day_sin','day_cos', 'week_sin', 'week_cos']


def get_X_columns(features = FEATURES, L = alt_lags):
    '''
    Returns the names of the X-columns for cross-sectional data. 
    '''
    lag_features = [f for f in features if f not in dont_lag]
    feature_columns = features.copy()
    for lag in L: # L is list in this setting
        for f in lag_features:
            name = f'{f}-{lag}'
            feature_columns.append(name)
    return feature_columns


def gen_cross_X(df, features = FEATURES, H = HORIZON, L = alt_lags, D = DARK_PATCH,df_timedate = None):
    '''
    Use this function to generate prediction data for use case. 
    '''
    lag_features = [f for f in features if f not in dont_lag]
    feature_columns = features.copy()
    data = df.copy()
    for lag in L: # L is list in this setting
        for f in lag_features:
            name = f'{f}-{lag}'
            data[name] = data[f].shift(lag)
            feature_columns.append(name)
    data = data.dropna()
    if len(data) > 0:
        success = True
        date_map = df_timedate[data.index].values
        X = np.array(data[feature_columns])
    else:
        success = False
        date_map = None
        X = None
    return success, X,date_map


def td_1d(df, features = FEATURES, H = HORIZON, L = alt_lags, D = DARK_PATCH):
    '''
    Generates cross-sectional structure observations from a single EV time series.
    The sliding window applied omitts all observations with a single NaN value and allows overlapping observations.
    The input data is the past 24 hours and the forecasted day one week prior. 

    Returns the X and Y datasets for training, validation and test.
    '''
    targets = ['soc','is_home'] 
    lag_features = [f for f in features if f not in dont_lag]
    feature_columns = features.copy()
    target_columns = []
    data = df.copy()
    for lag in L: # L is list in this setting
        for f in lag_features:
            name = f'{f}-{lag}'
            data[name] = data[f].shift(lag)
            feature_columns.append(name)
    for t in targets:
        for shift in range(D,D+H+1):
            name = f'{t}+{shift}'
            data[name] = data[t].shift(-shift)
            target_columns.append(name)
    data = data.dropna()
    
    X = np.array(data[feature_columns])
    Y = np.array(data[target_columns])

    n = Y.shape[0]
    break_val = int(n*0.6)
    break_test = int(n*0.8)

    X_train = X[:break_val,:]
    X_val = X[break_val:break_test,:]
    X_test = X[break_test:,:]

    Y_train = Y[:break_val,:]
    Y_val = Y[break_val:break_test,:]
    Y_test = Y[break_test:,:]

    return X_train,X_val,X_test,Y_train,Y_val,Y_test


def aggregate_td(files, features = FEATURES, H = HORIZON, L = alt_lags):
    '''
    Iterates the processed EV time series and generates cross-sectional structure datasets for training, validation and testing.
    Time is encoded and sin-cos signals, odometer is differentiated and categorical varibles are dummy-encoded.

    Returns the aggregated datasets for X and Y for traning, validation and test data.
    '''    
    c = len(features) + len(alt_lags)*(len(features)-len(dont_lag))
    X_train = np.empty(shape=(0,c))
    X_val = np.empty(shape=(0,c))
    X_test = np.empty(shape=(0,c))
    b = (H+1)*2
    Y_train = np.empty(shape=(0,b)) 
    Y_val = np.empty(shape=(0,b))
    Y_test = np.empty(shape=(0,b))
    
    for ind, file in enumerate(files):
        print(f'Time series processed: {round(ind / len(files) * 100, 1)} % Next: {file}')
        df = pd.read_csv(f'{datadir}/{file}')

        df = create_dummies(df)
        df = diff_odometer(df)
        df = encode_time(df) 

        x_train,x_val,x_test,y_train,y_val,y_test = td_1d(df, features, H, L)
        X_train = np.concatenate([X_train,x_train])
        X_val = np.concatenate([X_val,x_val])
        X_test = np.concatenate([X_test,x_test])

        Y_train = np.concatenate([Y_train,y_train])
        Y_val = np.concatenate([Y_val,y_val])
        Y_test = np.concatenate([Y_test,y_test])

    return X_train,X_val,X_test,Y_train,Y_val,Y_test            


def main():
    '''
    Reads the files from processed data directory, generates datasets and index mapping, and saves the generated data.
    '''
    print(f'Generating cross-sectional data with config:\n  FCR: {FCR}\n  DATE: {DATE}')

    files = os.listdir(datadir)
    X_train,X_val,X_test, Y_train,Y_val,Y_test = aggregate_td(files)

    print('X_train shape:',X_train.shape)
    print('Y_train shape:',Y_train.shape)
    
    n = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f'No. observations (full dataset, with week lag): n = {n}')

    np.save(f'{savedir}/X_train.npy',X_train)
    np.save(f'{savedir}/X_val.npy',X_val)
    np.save(f'{savedir}/X_test.npy',X_test)

    np.save(f'{savedir}/Y_train.npy',Y_train)
    np.save(f'{savedir}/Y_val.npy',Y_val)
    np.save(f'{savedir}/Y_test.npy',Y_test)


if __name__ == '__main__':
    main()