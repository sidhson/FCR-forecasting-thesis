import numpy as np
import os
import sys
import wandb
import yaml

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

DATE = '2022-05-16'
FCR = '1-D'
H = 49
save = True

datadir = f'{root}/data/train_{FCR}/{DATE}/cross_data'

from models.xgbforecaster import * # NOTE
from preprocessing.feature_engineering import normalize_2D_data_notY
from preprocessing.gen_cross_data import get_X_columns
from evaluate_models.model_scoring import *

all_cols = get_X_columns()


print('### Loading data ###')
X_train = np.load(f'{datadir}/X_train.npy')
X_val = np.load(f'{datadir}/X_val.npy')
X_test = np.load(f'{datadir}/X_test.npy')
Y_train = np.load(f'{datadir}/Y_train.npy')
Y_val = np.load(f'{datadir}/Y_val.npy')


X_train,X_val,X_test= normalize_2D_data_notY(X_train, X_test,X_val=X_val)


print('### Initializing model ###')
def train(sweeping = True):

    if not sweeping:
        wandb.init(project=f'XGB-project-{FCR}',entity = 'g_and_f')
        fn = f'{os.getcwd()}/best_config.yaml'
        with open(fn) as fh:
            config_file = yaml.load(fh, Loader=yaml.FullLoader)
            wandb.config = config_file['parameters']
        meta_config = wandb.config
    else:
        print('### Trying to initate Sweep ###')
        wandb.init()
        meta_config = wandb.config
        print('### Sweep initiated successfully ###')

    reg_config = {}
    bin_config = {}

    for k,v in meta_config.items():
        if k[:3] == 'reg':
            reg_config['_'.join(k.split('_')[1:])] = v
        elif k[:3] == 'bin':
            bin_config['_'.join(k.split('_')[1:])] = v
        else:
            print('incorrectly named meta configs in wandb')
            raise Exception()

    print('KWARGS SORTED OUT, INITIALIZING OBJECT')
    model = XGBForecaster(reg_config,bin_config)

    print('### Fitting model ###')
    model.fit(X_train,Y_train,verbose = False)

    reg_train,bin_train = model.predict(X_train)

    rmses_train = stats_per_horizon(Y_train[:,:H],reg_train,task = 'rmse')
    accuracies_train = stats_per_horizon(Y_train[:,H:],bin_train,task = 'clf',thresholds=None)

    reg_preds,bin_preds = model.predict(X_val)
    rmses = stats_per_horizon(Y_val[:,:H],reg_preds,task = 'rmse')
    accuracies = stats_per_horizon(Y_val[:,H:],bin_preds,task = 'clf',thresholds=None)

    wandb.log({'mean val rmse':np.mean(rmses),'mean val acc':np.mean(accuracies), 
                'mean train rmse':np.mean(rmses_train), 'mean train acc':np.mean(accuracies_train),
                'all val rmse': rmses, 'all val acc': accuracies})
    wandb.finish()
    if save:
        print('### Saving model ###')
        savedir = f'{root}/models_save/{FCR}/XGB'
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        model.save_self(savedir)


if __name__ == '__main__':
    train(sweeping=False)