'''
Generates sequential data for RNN models. 
Saves training, validation and testing datasets with splits (0.6, 0.2, 0.2). 
The sliding window used to generate the data takes the past 24 hours and the forecasted day one week prior as input. 
Additionally, saves an index mapping from observation to EV id and time-point.

NOTE: DO NOT FORGET to set correct DATE and FCR before the script is run! 
'''

import numpy as np 
import os
import pandas as pd
import sys

DATE = '2022-05-16'
FCR = '1-D'

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)
datadir = f'{root}/data/processed_data/data_{DATE}'
rawdir = f'{root}/data/raw_data/data_{DATE}'

from preprocessing.feature_engineering import *

if FCR == '1-D':
    HORIZON = int(48)
    DARK_PATCH = int(14) # Forecast created at 17.00
    LOOK_BACK = int(48)
elif FCR == '2-D':
    HORIZON = int(48)
    DARK_PATCH = int(20 + 48) # Forecast created at 14.00
    LOOK_BACK = int(48)

savedir = f'{root}/data/train_{FCR}/{DATE}/seq_data'
if not os.path.exists(savedir):
    os.mkdir(savedir)

FEATURES = ['day_sin', 'day_cos', 'week_sin', 'week_cos', 'odometer_diff', 'range', 'latitude', 'longitude', 'distance', 'isPluggedIn', 'charge_status_CHARGING', 'charge_status_FULLY_CHARGED', 'charge_status_NOT_CHARGING', 'soc', 'is_home']

def gen_X(df, features = FEATURES, H = HORIZON, L = LOOK_BACK,D = DARK_PATCH, df_timedate = None, ev_id = 0):
    '''
    Generates sequential structure observations from a single EV time series.
    The sliding window applied omitts all observations with a single NaN value and allows overlapping observations.
    The input data is the past 24 hours and the forecasted day one week prior. 

    Returns an indicator of whether any observations were created at all for the EV time series, along with the X_week, X_yest, and Y datasets for training, validation and test, and lastly the index_mapping arrays.
    '''
    start_shift = D-7*24*2 # First datetime of input data one week ago.
    X_week = []
    X_yest = []
    ind_map = []
    n = len(df)
    t = -1*start_shift # Set first index. 
    while t < n: 
        # loc used for slicing. Includes endpoints. 
        chunk_week = df.loc[t+start_shift:t+start_shift+H,features].copy()
        chunk_yest = df.loc[t-L:t,features].copy()
        
        # Check NaNs per chunk used. 
        nan_week = np.where(chunk_week.isnull().any(axis = 1))[0]
        nan_yest = np.where(chunk_yest.isnull().any(axis = 1))[0]

        # Not a single NaN is allowed. 
        if len(nan_week) + len(nan_yest)  == 0:
            X_yest.append(np.array(chunk_yest))
            X_week.append(np.array(chunk_week))

            ind_map.append(np.array([df_timedate[t], int(ev_id)])) # Map timedate from last X timepoint.
            
            t += 1
        else:
            m_week = 0
            m_yest = 0
            if len(nan_week) != 0: m_week = max(nan_week)
            if len(nan_yest) != 0: m_yest = max(nan_yest)
            t += max(m_week,m_yest)+1

    if len(X_week) > 0:
        success = True # Check if time series contains any usable data at all. 
        X_week = np.stack(X_week)
        X_yest = np.stack(X_yest)
        ind_map = np.stack(ind_map)
        
    else:
        success = False
        X_week = None
        X_yest = None
        ind_map = None

    return success, X_week, X_yest, ind_map



def new_td(df, features = FEATURES, H = HORIZON, L = LOOK_BACK,D = DARK_PATCH, df_timedate = None, ev_id = 0):
    '''
    Generates sequential structure observations from a single EV time series.
    The sliding window applied omitts all observations with a single NaN value and allows overlapping observations.
    The input data is the past 24 hours and the forecasted day one week prior. 

    Returns an indicator of whether any observations were created at all for the EV time series, along with the X_week, X_yest, and Y datasets for training, validation and test, and lastly the index_mapping arrays.
    '''
    start_shift = D-7*24*2 # First datetime of input data one week ago.
    X_week = []
    X_yest = []
    Y = []
    ind_map = []
    n = len(df)
    t = -1*start_shift # Set first index. 
    while t < n-H-D: 
        # loc used for slicing. Includes endpoints. 
        chunk_week = df.loc[t+start_shift:t+start_shift+H,features].copy()
        chunk_yest = df.loc[t-L:t,features].copy()
        chunk_Y = df.loc[t+D:t+D+H,['soc','is_home']].copy()
        
        # Check NaNs per chunk used. 
        nan_week = np.where(chunk_week.isnull().any(axis = 1))[0]
        nan_yest = np.where(chunk_yest.isnull().any(axis = 1))[0]
        nan_Y = np.where(chunk_Y.isnull().any(axis = 1))[0]

        # Not a single NaN is allowed. 
        if len(nan_week) + len(nan_yest) + len(nan_Y) == 0:
            X_yest.append(np.array(chunk_yest))
            X_week.append(np.array(chunk_week))
            Y.append(np.array(chunk_Y))

            ind_map.append(np.array([df_timedate[t], int(ev_id)])) # Map timedate from last X timepoint.
            
            t += 1
        else:
            m_week = 0
            m_yest = 0
            m_Y = 0
            if len(nan_week) != 0: m_week = max(nan_week)
            if len(nan_yest) != 0: m_yest = max(nan_yest)
            if len(nan_Y) != 0: m_Y = max(nan_Y)
            t += max(m_week,m_yest,m_Y)+1

    if len(Y) > 0:
        success = True # Check if time series contains any usable data at all. 
        X_week = np.stack(X_week)
        X_yest = np.stack(X_yest)
        Y = np.stack(Y)
        ind_map = np.stack(ind_map)
        
        n = Y.shape[0]
        break_val = int(n*0.6)
        break_test = int(n*0.8)
        
        X_week_train = X_week[:break_val,:,:]
        X_week_val = X_week[break_val:break_test,:,:]
        X_week_test = X_week[break_test:,:,:]

        X_yest_train = X_yest[:break_val,:,:]
        X_yest_val = X_yest[break_val:break_test,:,:]
        X_yest_test = X_yest[break_test:,:,:]

        Y_train = Y[:break_val,:,:]
        Y_val = Y[break_val:break_test,:,:]
        Y_test = Y[break_test:,:,:]

        ind_map_train = ind_map[:break_val]
        ind_map_val = ind_map[break_val:break_test]
        ind_map_test = ind_map[break_test:]
    else:
        success = False
        X_week_train = None 
        X_week_val = None
        X_week_test = None

        X_yest_train = None
        X_yest_val = None
        X_yest_test = None

        Y_train = None
        Y_val = None
        Y_test = None

        ind_map_train = None
        ind_map_val = None
        ind_map_test = None

    return success,X_week_train,X_week_val,X_week_test,X_yest_train,X_yest_val,X_yest_test,Y_train,Y_val,Y_test,ind_map_train,ind_map_val,ind_map_test
   

def aggregate_td(files, features = FEATURES, H = HORIZON, L = LOOK_BACK, D = DARK_PATCH):        
    '''
    Iterates the processed EV time series and generates sequential structure datasets for training, validation and testing.
    Additionally, generates index mapping from observation to EV id and time-point.
    Time is encoded and sin-cos signals, odometer is differentiated and categorical varibles are dummy-encoded.

    Returns the aggregated datasets for X_week, X_yest and Y for traning, validation and test data, along with the index mappings.
    '''
    X_week_train = np.empty(shape=(0,H+1,len(features)))
    X_week_val = np.empty(shape=(0,H+1,len(features)))
    X_week_test = np.empty(shape=(0,H+1,len(features)))
    
    X_yest_train = np.empty(shape=(0,L+1,len(features)))
    X_yest_val = np.empty(shape=(0,L+1,len(features)))
    X_yest_test = np.empty(shape=(0,L+1,len(features)))

    Y_train = np.empty(shape=(0,H+1,2)) # Two target variables.
    Y_val = np.empty(shape=(0,H+1,2))
    Y_test = np.empty(shape=(0,H+1,2))

    # Variables for mapping observation --> EV id and time-point. 
    ind_map_train = np.empty(shape=(0,2))
    ind_map_val = np.empty(shape=(0,2))
    ind_map_test = np.empty(shape=(0,2)) 

    for ind, file in enumerate(files):
        print(f'Time series processed: {round(ind / len(files) * 100, 1)} % Next: {file}')

        df = pd.read_csv(f'{datadir}/{file}')
        ev_id = file.split('_')[-1].rstrip('.csv')
        df_timedate = pd.to_datetime(df['time']).astype(int)/ 10**9 # Reverse by pd.to_datetime(ind_map[obs,2], unit='s') 

        df = create_dummies(df)
        df = diff_odometer(df)
        df = encode_time(df) # Make sure time is encoded last. 
                                                                                      
        success,x_week_train,x_week_val,x_week_test,x_yest_train,x_yest_val,x_yest_test,y_train,y_val,y_test,ind_map_train_i,ind_map_val_i,ind_map_test_i = new_td(df,features,H,L,D,df_timedate,ev_id)
        
        if success: # Only concatenate data for EVs which contain useable data. 
            X_week_train = np.concatenate([X_week_train,x_week_train])
            X_week_val = np.concatenate([X_week_val,x_week_val])
            X_week_test = np.concatenate([X_week_test,x_week_test])

            X_yest_train = np.concatenate([X_yest_train,x_yest_train])
            X_yest_val = np.concatenate([X_yest_val,x_yest_val])
            X_yest_test = np.concatenate([X_yest_test,x_yest_test])

            Y_train = np.concatenate([Y_train,y_train])
            Y_val = np.concatenate([Y_val,y_val])
            Y_test = np.concatenate([Y_test,y_test])

            ind_map_train = np.concatenate([ind_map_train,ind_map_train_i])
            ind_map_val = np.concatenate([ind_map_val,ind_map_val_i])
            ind_map_test = np.concatenate([ind_map_test,ind_map_test_i])
            
    return X_week_train,X_week_val,X_week_test,X_yest_train,X_yest_val,X_yest_test,Y_train,Y_val,Y_test,ind_map_train,ind_map_val,ind_map_test


def main():
    '''
    Reads the files from processed data directory, generates datasets and index mapping, and saves the generated data.
    '''
    print(f'Generating sequential data with config:\n  FCR: {FCR}\n  DATE: {DATE}')

    files = os.listdir(datadir)
    X_week_train,X_week_val,X_week_test,X_yest_train,X_yest_val,X_yest_test,Y_train,Y_val,Y_test,ind_map_train,ind_map_val,ind_map_test = aggregate_td(files)
    print('X_yest_train shape:',X_yest_train.shape)
    print('X_week_train shape:',X_week_train.shape)
    print('Y_train shape:',Y_train.shape)
    print(f'Ind_map shape: {ind_map_train.shape}')

    n = X_yest_test.shape[0] + X_yest_val.shape[0] + X_yest_train.shape[0]
    print(f'No. observations (full dataset, with week): n = {n}')

    np.save(f'{savedir}/X_week_train.npy',X_week_train)
    np.save(f'{savedir}/X_week_val.npy',X_week_val)
    np.save(f'{savedir}/X_week_test.npy',X_week_test)

    np.save(f'{savedir}/X_yest_train.npy',X_yest_train)
    np.save(f'{savedir}/X_yest_val.npy',X_yest_val)
    np.save(f'{savedir}/X_yest_test.npy',X_yest_test)

    np.save(f'{savedir}/Y_train.npy',Y_train)
    np.save(f'{savedir}/Y_val.npy',Y_val)
    np.save(f'{savedir}/Y_test.npy',Y_test)

    np.save(f'{savedir}/ind_map_train.npy',ind_map_train)
    np.save(f'{savedir}/ind_map_val.npy',ind_map_val)
    np.save(f'{savedir}/ind_map_test.npy',ind_map_test)


if __name__ == '__main__':
    main()