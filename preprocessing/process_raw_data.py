'''
Processes the raw data of EV time series and saves them to processed data. 

For each EV time series: resamples to 30 minutes, interpolates and computes `distance` and `is_home` attributes.
'''

import geopy.distance
import numpy as np
import os
import pandas as pd
import sys
from scipy import interpolate

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from data.check_ev_location import evs_not_recording_location

DATE = '2022-05-16'
datadir_raw = f'{root}/data/raw_data/data_{DATE}'
ev_info = pd.read_csv(f'{datadir_raw}/EV_constants.csv')
files = os.listdir(datadir_raw)

outdir = f'{root}/data/processed_data/data_{DATE}'
if not os.path.exists(outdir):
    os.mkdir(outdir)


def get_min_diffs(df):
    '''
    Computes the minute differences between subsequent records in the EV time series.
    
    Returns a Numpy array with the differences in minutes.
    '''
    min_diffs = []
    for i in range(1,len(df)):
        delta = df['time'][i]-df['time'][i-1]
        min_diffs.append(delta.days*24*60+delta.seconds/60)
    min_diffs = np.array(min_diffs)
    return min_diffs


def get_distances(df,id):
    '''
    Computes and adds the `distance` to the home location for the EV time series.

    Returns the EV time series with the `distance` attribute. 
    '''
    home = ev_info['home_location'][ev_info['id']==int(id)].values
    home_lat,home_long = list(map(float,home[0].lstrip('(').rstrip(')').split(',')))
    distances = []
    for i in range(len(df)):
        try:
            dist = geopy.distance.geodesic((df['latitude'][i],df['longitude'][i]),(home_lat,home_long)).m
        except:
            dist = np.nan
        distances.append(dist)
    return distances


def get_is_home(df):
    '''
    Computes and adds the `is_home` attribute of an EV time sereis. 
    Assumes a distance to the home location of 150 m.

    Returns the EV time series with the `is_home` attribute.
    '''
    is_home = []
    for _, row in df.iterrows():
        if np.isnan(row['distance']):
            is_home.append(np.nan)
        else:
            is_home.append(row['distance'] <= 150)
    return is_home


def interpolate_single_categorical(last_soc, next_soc, distance):
    ''' 
    Interpolates categorical variables 'charge_status' and 'isPluggedIn' for single rows.
    Returns a tuple of the interpolated values as 'plugged_i', 'charge_i'. 
    '''
    if (not np.isnan(last_soc)) and (not np.isnan(next_soc)) and (next_soc > last_soc):
        plugged_i = True
        charge_i = 'CHARGING'
    else:
        plugged_i = np.nan
        charge_i = np.nan
    
    return charge_i, plugged_i


def is_nan(input, col_type):
    '''Returns True if the input is a NaN value based on some col_type.'''
    if col_type == 'float':
        return np.isnan(input)
    elif col_type == 'bool':
        return not isinstance(input,bool)
    elif col_type == 'charge_status':
        return input not in ['CHARGING', 'NOT_CHARGING', 'FULLY_CHARGED']


def mask_nan(s, col_type, max_consec):
    '''
    Creates a mask for a column of which values should be NaNs. 
    Sets True if value should be kept, False for NaN.
    Assumes interpolation of 5 consecutives NaN values.
    '''
    mask = np.array([False]*len(s))
    mask[0] = not is_nan(s[0],col_type)
    i = 1
    while i < len(s):
        went_in = False
        if is_nan(s[i],col_type):
            if mask[i-1]:
                j = i+1
                went_in = True
                while j < len(s) and is_nan(s[j],col_type):
                    j+=1
                if j-i <= max_consec:
                    mask[i:j] = True
                    
        if went_in:
            i=j
        else:
            mask[i] = True
            i+=1
    return mask


def pad_missing_timesteps(df):
    '''
    Fills large gaps of missing timesteps (over 50 minutes) by padding with adjecent sampling frequencies.
    '''
    min_diffs = get_min_diffs(df)
    inds = np.where(min_diffs>50)[0] # For index i, padding performed between i and i+1.

    while len(inds)>0:
        i = inds[0]
        d1 = df.time[i]
        d2 = df.time[i+1]
        delta = d2-d1
        missing = delta.days*24*60 + delta.seconds/60 # Missing time steps.
        freq = int(min_diffs[i-1])
        insdf = pd.DataFrame(columns = df.columns)
        d = pd.Timedelta(minutes=freq)
        r = pd.date_range(start=d1+d, end=d2-d, periods = (round((missing)/freq,0)-1))
        insdf['time'] = r
        for c in insdf.columns:
            if c != 'time':
                insdf[c] = np.nan
        df = pd.concat([df[:i+1],insdf,df[i+1:]],ignore_index = True)

        min_diffs = get_min_diffs(df)
        inds = np.where(min_diffs>50)[0] # For index i, padding performed between i and i+1.
    return df


def _set_nan(a, b, row_time):
    ''' Returns true if row value should be set to NaN.'''
    a_time, a_bool = a
    b_time, b_bool = b
    if not (a_bool or b_bool):
        return True
    elif a_bool and not b_bool and row_time-a_time > 15*60:
        return True
    elif b_bool and not a_bool and b_time-row_time > 15*60:
        return True
    return False


def create_new_freq_df(df):
    '''
    Creates a new dataframe which is correctly sampled at 30 min. 
    
    Returns the newly sampled dataframe and the old dataframe's timestamps in UNIX time (seconds).
    '''
    start_time = df.loc[0, 'time']
    start_time = start_time.round(freq='30T') + pd.Timedelta(minutes = 30) # Makes sure start time is in bounds.
    t_new = pd.date_range(start=start_time, end=df.loc[len(df)-1, 'time'], freq='30T')
    df_new = pd.DataFrame()
    df_new['time'] = t_new

    t_new_num = pd.to_numeric(df_new['time']) / 10**9 # Numeric values in seconds.
    t_old_num = pd.to_numeric(df['time']) / 10**9
    df_new['time_num'] = t_new_num
    return df_new, t_old_num


def interpolate_continuous(df, df_new, t_old_num,max_consec):
    '''
    Interpolates continuous variables up to k=5 steps. Linear interpolation independent of sampling frequency (20,30,40).
    Keeps NaN values for non-interpolated timesteps. 

    Returns the new interpolated dataframe and the mask indicating continuous NaN values. 
    '''
    cont_cols = ['soc','odometer','range','latitude','longitude']
    for c in cont_cols:
        not_null = df.index[df[c].notnull()] # Cannot give NaN values to interp1d.
        inter = interpolate.interp1d(t_old_num[not_null], df.loc[not_null,c], bounds_error=False) 
        df_new[c] = inter(df_new['time_num'])

    ## Create a mask for which rows shuold be interpolated. 
    mask = pd.DataFrame()
    mask['time'] = df['time']
    mask['time_num'] = t_old_num
    for c in cont_cols:
        mask[c] = mask_nan(df[c],col_type='float',max_consec=max_consec)
    mask['not_nan'] = mask.loc[:,cont_cols].all(axis=1) # Keep row if all numerical variables should be interpolated.
    mask = mask.drop(cont_cols, axis=1)

    ## Set NaN values for regions which should not be interpolated. 
    for i, row in df_new.iterrows():
        row_time = row['time_num']
        # NOTE: More efficient implementation? One take the first value found and not the whole vector. 
        a = mask[['time_num','not_nan']][mask['time_num'] <= row_time].iloc[-1,:]
        b = mask[['time_num','not_nan']][mask['time_num'] >= row_time].iloc[0,:]

        # Extrapolates up to 15 minutes and interpolates intervals of up to 40 minutes. 
        if _set_nan(a, b, row_time):
            df_new.loc[i, cont_cols] = np.nan
    return df_new, mask


def interpolate_categorical(df, df_new, t_old_num, mask,max_consec):
    '''
    Interpolates categorical variables by (1) setting NaN if numerical variables indicate NaN-row, 
    (2) using sophisticated interpolation (see: 'interpolate_single_categorical'), or (3) using the 
    interpolated value with 'nearest' method.

    Returns the new dataframe with interpolated categorical variables.
    '''
    cat_cols = ['charge_status', 'isPluggedIn']
    charge_map_to_int = {'CHARGING':0,'NOT_CHARGING':1,'FULLY_CHARGED':2}
    plug_map_to_int = {False:0,True:1}
    charge_map_to_str = {v:k for k,v in charge_map_to_int.items()}
    plug_map_to_bool = {v:k for k,v in plug_map_to_int.items()}

    mask_cat = pd.DataFrame()
    mask_cat['time'] = df['time']
    mask_cat['time_num'] = t_old_num
    
    mask_cat['isPluggedIn'] = mask_nan(df['isPluggedIn'], col_type='bool',max_consec=max_consec)
    mask_cat['charge_status'] = mask_nan(df['charge_status'], col_type='charge_status',max_consec=max_consec)
    mask_cat['not_nan'] = mask_cat.loc[:,cat_cols].all(axis=1) # Keep row if all numerical variables should be interpolated.
    mask_cat = mask_cat.drop(cat_cols, axis=1)
    # Categorical 'isPluggedIn' and 'charge_status' are missing at the same time for some data, these are interpolated simultaneously.

    c_tmp = df['charge_status'].map(charge_map_to_int)
    not_null = df.index[c_tmp.notnull()] # Cannot give NaN values to interp1d.
    inter = interpolate.interp1d(t_old_num[not_null], c_tmp[not_null], kind='nearest', bounds_error=False)
    df_new['charge_status'] = inter(df_new['time_num'])

    c_tmp = df['isPluggedIn'].map(plug_map_to_int)
    not_null = df.index[c_tmp.notnull()] # Cannot give NaN values to interp1d.
    inter = interpolate.interp1d(t_old_num[not_null], c_tmp[not_null], kind='nearest', bounds_error=False)
    df_new['isPluggedIn'] = inter(df_new['time_num'])
    
    for i, row in df_new.iterrows():
        row_time = row['time_num']
        a_cont = mask[['time_num','not_nan']][mask['time_num'] <= row_time].iloc[-1,:]
        b_cont = mask[['time_num','not_nan']][mask['time_num'] >= row_time].iloc[0,:]
        
        # (1) If numerical mask is NaN, then whole row should be NaN.
        if _set_nan(a_cont, b_cont, row_time):
            df_new.loc[i, cat_cols] = np.nan
        else:
            a_cat = mask_cat[['time_num','not_nan']][mask_cat['time_num'] <= row_time].iloc[-1,:]
            b_cat = mask_cat[['time_num','not_nan']][mask_cat['time_num'] >= row_time].iloc[0,:]
        # (2) If there are num. values, but no cat., then use sophisticated interpolation. 
            if _set_nan(a_cat, b_cat, row_time):
                # df_new.loc[i, cat_cols] = np.nan
                if i not in [0, len(df_new)-1]: 
                    last_soc = df_new.loc[i-1, 'soc']
                    next_soc = df_new.loc[i+1, 'soc']
                    charge_i, plug_i = interpolate_single_categorical(last_soc, next_soc, row.distance)
                    if not np.isnan(plug_i):
                        df_new.loc[i, cat_cols] = charge_map_to_int[charge_i], plug_map_to_int[plug_i]
                    else:
                        df_new.loc[i,cat_cols] = np.nan
                else:
                    df_new.loc[i,cat_cols] = np.nan
        # (3) If the row contains numerical values, and categorical closeby, 
        # then use the interpolated nearest value already set.

    df_new.loc[:,'charge_status'] = df_new.loc[:,'charge_status'].map(charge_map_to_str)
    df_new.loc[:,'isPluggedIn'] = df_new.loc[:,'isPluggedIn'].map(plug_map_to_bool)

    return df_new

def process_raw_data_df(df,id,max_consec=5):
    ''''
    Processes single raw time series of an EV. Changes the sampling frequency to 30 minutes and linearly interpolates continuous variables and a combination of sopisticated/nearest interpolation of categorical variables. 

    Returns dataframe with additional variables 'distance' (to home location) and binary variable for EV plugged in at home. 
    '''
    ## Update format. Converts time to pandas datetime format (UTC). 
    df = df.drop('period',axis = 1)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    ## Fill in the missing gaps of in the time dimension (from server failure).
    df = pad_missing_timesteps(df)

    ## Create new dataframe with 30 min sampling frequency.
    df_new, t_old_num = create_new_freq_df(df)

    ## Interpolate all numeric variables. 
    df_new, mask = interpolate_continuous(df, df_new, t_old_num,max_consec)

    ## Add distance
    df_new['distance'] = get_distances(df_new,id)

    ## Interpolate categorical variables. 
    df_new = interpolate_categorical(df, df_new, t_old_num, mask,max_consec)

    ## Add is_home
    df_new['is_home'] = get_is_home(df_new)

    df_new = df_new.drop('time_num', axis=1) # Only used for interpolation. 
    return df_new


def main():
    '''
    Iterates all EV time series, processes the raw data and saves it to `processed_data`.
    '''
    omit_evs = evs_not_recording_location(date=DATE)
    for ind, file in enumerate(files):
        print(f'Files processed: {round(ind / len(files) * 100, 1)} %')
        if 'time' in file:
            id = file.split('.')[0].split('_')[-1]
            if id not in omit_evs:
                print(f'Next EV to process: {id}\n')
                df = pd.read_csv(f'{datadir_raw}/{file}')
                df = process_raw_data_df(df,id)
                df.to_csv(f'{outdir}/time_series_{id}.csv', index=False)


if __name__ == '__main__':
    main()