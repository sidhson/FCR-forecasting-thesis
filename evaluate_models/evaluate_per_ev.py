'''
Computes statistics per EV.
'''
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import sys

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from evaluate_models.model_evaluation import Evaluate
from evaluate_models.model_scoring import *

model_names = ['naive_week','coFF-oFF','coFF-oREC','XGB']

FCR = '1-D'
date = '2022-05-16'
data_set = 'test'

ind_map = np.load(f'{root}/data/train_{FCR}/{date}/seq_data/ind_map_{data_set}.npy')
inds = pd.DataFrame(ind_map, columns = ['time','EV'])
inds['time'] = pd.to_datetime(inds['time'],utc=True,unit = 's')
inds['EV'] = [int(id) for id in inds['EV']]
ev_ids = inds['EV'].unique()


def calculate_stats_per_ev(eval):
    '''
    Computes statistics for each EV and stores in a DataFrame. 
    RMSE, MAE, accuracy and F1-score along with total number of observations per EV. 
    Stores tabular summary as CSV. 
    '''
    savedir = f'{root}/results/{FCR}/{data_set}/ev_comparison'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    stats_df = pd.DataFrame(columns = ['EV','RMSE', 'MAE', 'Acc', 'F1', 'obs'])

    for id in ev_ids:

        ev_stats = {'EV':id}
        ev_inds = np.where(inds['EV'] == id)[0]
        
        ## Regression stats
        true = eval.soc_data['true'][ev_inds,:]
        preds = eval.soc_data['preds'][ev_inds,:]
        stat = stats_per_horizon(true, preds, task='rmse')
        ev_stats['RMSE'] = np.mean(stat)
        stat = stats_per_horizon(true, preds, task='mae')
        ev_stats['MAE'] = np.mean(stat)

        ## Classification stats
        true = eval.home_data['true'][ev_inds]
        preds = eval.home_data['preds'][ev_inds]
        stat = stats_per_horizon(true, preds, task='clf')
        ev_stats['Acc'] = np.mean(stat)
        stat = get_f1score(true, preds)
        ev_stats['F1'] = np.mean(stat)

        ev_stats['obs'] = len(ev_inds)

        stats_df = stats_df.append(ev_stats, ignore_index=True)

    stats_df.to_csv(f'{savedir}/ev_comparison_{eval.model_name}.csv')
    print(stats_df)


def plot_hist(model_name):
    '''
    Saves RMSE histogram plots over the EVs. 
    '''
    readdir = f'{root}/results/{FCR}/{data_set}/ev_comparison'
    stats_df = pd.read_csv(f'{readdir}/ev_comparison_{model_name}.csv')

    fig = plt.figure(figsize=(10,6))
    plt.hist(stats_df['RMSE'], bins=50)
    plt.xlabel('RMSE')

    fig.savefig(f'{readdir}/hist_RMSE_{model_name}.png', dpi=300, bbox_inches = 'tight')


def main():
    # (1) Compute stats per EV. 
    evals = [Evaluate(model_name, data_set=data_set, load_data=True, fcr=FCR) for model_name in model_names]
    for eval in evals:
        calculate_stats_per_ev(eval)

    # (2) Plot hitsograms.
    # for model_name in model_names:
    #     plot_hist(model_name, FCR)


if __name__ == '__main__':
    main()