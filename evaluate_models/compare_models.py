import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from evaluate_models.model_scoring import *
from model_evaluation import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

FCR = '2-D'

MODELS_LIST = ['naive_week','coFF-oFF','coFF-oREC','XGB']
DATASET = 'test'
SAVE = True
CONF_INT = False

if DATASET == 'test':
    savedir = f'{root}/results/{FCR}/test'
elif DATASET == 'val':
    savedir = f'{root}/results/{FCR}/val'

if not os.path.exists(savedir):
    os.mkdir(savedir)
pref_figsize = (10,6)

evaluations = [Evaluate(model,data_set=DATASET,load_data=True,fcr=FCR) for model in MODELS_LIST]
performance_scores = pd.DataFrame(index=MODELS_LIST, columns=['RMSE','MAE','Acc','F1'])

def compare_rmse(evaluations,conf_int=False):
    fig = plt.figure(figsize=pref_figsize)
    for i, eval in enumerate(evaluations): 
        if conf_int:
            rmses_distr = distr_per_horizon(eval.soc_data['true'],eval.soc_data['preds'],task = 'rmse')
            plt.plot(rmses_distr[:,1],label= eval.model_name, color=f'C{i}', marker='.')
            plt.plot(rmses_distr[:,0],label= eval.model_name, color=f'C{i}', marker='.', ls='--')
            plt.plot(rmses_distr[:,2],label= eval.model_name, color=f'C{i}', marker='.', ls='--')
            performance_scores.loc[eval.model_name, 'RMSE'] = np.mean(rmses_distr[:,1])
        else:
            rmses = stats_per_horizon(eval.soc_data['true'],eval.soc_data['preds'],task = 'rmse')
            plt.plot(rmses,label= eval.disp_name, marker='.')
            performance_scores.loc[eval.model_name, 'RMSE'] = np.mean(rmses)

    if DATASET == 'val':
        plt.title(f'FCR {FCR} validation data')
    elif DATASET == 'test':
        plt.title(f'FCR {FCR} test data')
    plt.xlabel('Horizon')
    plt.ylabel('RMSE')
    plt.legend()
    
    if SAVE:
        fig.savefig(f'{savedir}/rmse.png', format='png')


def compare_mae(evaluations,conf_int=False):
    fig = plt.figure(figsize=pref_figsize)
    for i, eval in enumerate(evaluations): 
        if conf_int:
            mae_distr = distr_per_horizon(eval.soc_data['true'],eval.soc_data['preds'],task = 'mae')
            plt.plot(mae_distr[:,1],label= f'{eval.model_name} mean', color=f'C{i}', marker='o')
            plt.plot(mae_distr[:,2],label= f'{eval.model_name} 0.975p', color=f'C{i}', marker='.', ls='--')
            plt.plot(mae_distr[:,0],label= f'{eval.model_name} 0.025p', color=f'C{i}', marker='.', ls='--')
            performance_scores.loc[eval.model_name, 'MAE'] = np.mean(mae_distr[:,1])
        else:
            mae = stats_per_horizon(eval.soc_data['true'],eval.soc_data['preds'],task = 'mae')
            plt.plot(mae,label= eval.disp_name,  marker='.')
            performance_scores.loc[eval.model_name, 'MAE'] = np.mean(mae)

    if DATASET == 'val':
        plt.title(f'FCR {FCR} validation data')
    elif DATASET == 'test':
        plt.title(f'FCR {FCR} test data')
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    plt.legend()
    
    if SAVE:
        fig.savefig(f'{savedir}/mae.png', format='png')


def compare_accuracy(evaluations,conf_int=False):
    fig = plt.figure(figsize=pref_figsize)
    for i, eval in enumerate(evaluations):
        accuracies = stats_per_horizon(eval.home_data['true'],eval.home_data['preds'],task = 'clf',thresholds=None)
        performance_scores.loc[eval.model_name, 'Acc'] = np.mean(accuracies)
        plt.plot(accuracies,label = eval.disp_name,  marker='.')
    if DATASET == 'val':
        plt.title(f'FCR {FCR} validation data')
    elif DATASET == 'test':
        plt.title(f'FCR {FCR} test data')
    plt.xlabel('Horizon')
    plt.ylabel('Accuracy')
    plt.legend()
    if SAVE:
        fig.savefig(f'{savedir}/accuracies.png',format = 'png')


def compare_f1(evaluations,conf_int=False):
    fig = plt.figure(figsize=pref_figsize)
    for eval in evaluations:
        f1 = get_f1score(eval.home_data['true'],eval.home_data['preds'],thresholds=None)
        performance_scores.loc[eval.model_name, 'F1'] = np.mean(f1)
        plt.plot(f1,label = eval.disp_name, marker='.')
    if DATASET == 'val':
        plt.title(f'FCR {FCR} validation data')
    elif DATASET == 'test':
        plt.title(f'FCR {FCR} test data')
    plt.xlabel('Horizon')
    plt.ylabel('F1-score')
    plt.legend()
    if SAVE:
        fig.savefig(f'{savedir}/f1_scores.png',format = 'png')


def violin_plots(evaluations, horizons=[1, 25, 49]):
    '''
    Generates violin plots of the error distributions at given forecasted horizons. 
    '''
    ### (1) SAME FIGURE WITH SUBPLOTS.
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=pref_figsize)
    for plt_i, h in enumerate(horizons):
        model_names = []
        for i, eval in enumerate(evaluations):
            true = eval.soc_data['true']
            preds = eval.soc_data['preds']
            errors = true[:,h-1] - preds[:,h-1]
            ax[plt_i].violinplot(dataset=errors, positions=[i])
            model_names.append(eval.disp_name)
        ax[plt_i].set_xticks(range(len(model_names)))
        ax[plt_i].set_xticklabels(model_names)
        ax[plt_i].set_title(f'Horizon {h}')
        if DATASET == 'val':
            plt.suptitle(f'FCR {FCR} validation data')
        elif DATASET == 'test':
            plt.suptitle(f'FCR {FCR} test data')

        fig.savefig(f'{savedir}/violin_all.png',format = 'png')


if __name__ == '__main__':
    compare_rmse(evaluations, conf_int=CONF_INT)
    compare_mae(evaluations, conf_int=CONF_INT)
    compare_accuracy(evaluations, conf_int=CONF_INT)
    compare_f1(evaluations, conf_int=CONF_INT)

    violin_plots(evaluations)

    performance_scores.to_csv(f'{root}/results/{FCR}/{DATASET}/performance_tab.csv')
