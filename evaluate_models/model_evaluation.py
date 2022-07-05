'''
Run this script to generate and save prediction data for each model. 
Also generates individual preformance plots for the models. 
The Evaluation class is used to carry prediction data on the model, this is in turn used 
to compare the models in different aspects. 

The plot-functions are labeled accordingly to what is plotted (F1,RMSE,MAE and Accuracy) and is plotted 
over the forecasted timesteps. 
'''

import matplotlib.pyplot as plt
import numpy as np 
import os
from sklearn.metrics import roc_curve,roc_auc_score
import sys
import tensorflow as tf

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from evaluate_models.model_scoring import *
from preprocessing.feature_engineering import normalize_3D_data_notY, normalize_2D_data_notY
from models import xgbforecaster
from models.naive_model_week import * 

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

### CONFIGURE PERFORMANCE EVALUATION ###
DATE = '2022-05-16'
H = 49
pref_figsize = (10,6)

FCR = '2-D'
DATASET_LIST = ['val','test']
MODELS_LIST = ['naive_week','coFF-oFF','coFF-oREC','XGB']

class Evaluate:
    '''
    Objects of this class carry prediction data for specific model, FCR setting (1-D or 2-D) and 
    dataset (validation or test). 
    To save computing time, the raw predictions are saved along with true values, on the same format
    for all models. This is done by running this script as main, and for future creation of said objects, 
    this data can be loaded from files in stead of creating new predictions. 
    '''
    def __init__(self, model_name, data_set='val', load_data=True, fcr=FCR):
        self.model_name = model_name
        self.load_data = load_data
        self.data_set = data_set
        self.fcr = fcr

        print(f'### Loading model: {model_name} ###')
        self.model = self.load_model_and_set_dir()
        self.run_model()
    
    def load_model_and_set_dir(self):
        '''
        Sets the model name, display name in plots, loads the model and sets data directory.
        '''
        self.savedir = f'{root}/evaluate_models/prediction_data/{self.fcr}/{self.model_name}'
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
            
        if 'XGB' in self.model_name:
            read_dir = f'{root}/models_save/{self.fcr}/XGB'
            model = xgbforecaster.load_model(read_dir)
            self.disp_name = 'XGBoost'
            self.datadir = f'{root}/data/train_{self.fcr}/{DATE}/cross_data'
            print('XGB-model loaded.')
        elif 'naive' in self.model_name:
            model = NaiveWeek()
            self.disp_name = 'Naive baseline'
            self.datadir = f'{root}/data/train_{self.fcr}/{DATE}/seq_data'
            print('Naive model loaded.')
        else:
            readfile = f'{root}/models_save/{self.fcr}/{self.model_name}.h5'
            model = tf.keras.models.load_model(readfile)
            if self.model_name == 'coFF-oFF':
                self.disp_name = 'RNN-oFF'
            elif self.model_name == 'coFF-oREC':
                self.disp_name = 'RNN-oREC'
            self.datadir = f'{root}/data/train_{self.fcr}/{DATE}/seq_data'
            print('LSTM model loaded: ', model.summary())

        return model


    def run_model(self):
        '''
        If self.load_data == True, the object finds the prediction data saved from previous run. 
        If self.load_data == False, then the predictions must be generated from the models. 
        The main outcome is that predictions are stored separately for soc and is_home, and that the true
        data needs to be on a common format. For RNN models, Y has the shape (N,H,2), whereas for XGB the 
        Y data has the shape (N,2*H). The outcome is that soc data and is_home data is saved as (N,H) individually.  
        '''

        if not self.load_data:
            ## Compute predictions and save as .npy for next use.
            print('### Computing predictions... ###')
            self.soc_data, self.home_data = self.compute_predictions()
            for data_set_full, data in self.soc_data.items():
                if 'preds' in data_set_full:

                    np.save(f'{self.savedir}/soc_preds_{self.data_set}.npy',data) # Save per target var. and dataset.

            for data_set_full, data in self.home_data.items():
                if 'preds' in data_set_full:
                    np.save(f'{self.savedir}/home_preds_{self.data_set}.npy',data) # Save per target var. and dataset.

        else:
            self.soc_data = {}
            self.home_data = {}

            ## Load previous predictions on data set.
            self.soc_data['preds'] = np.load(f'{self.savedir}/soc_preds_{self.data_set}.npy')
            self.home_data['preds'] = np.load(f'{self.savedir}/home_preds_{self.data_set}.npy')
            
            ## Load true data
            true = self.read_data_Y(data_set=self.data_set)
            if 'XGB' in self.model_name:
                self.soc_data['true'] = true[:,:H]
                self.home_data['true'] = true[:,H:]
            else:
                self.soc_data['true'] = true[:,:,0]
                self.home_data['true'] = true[:,:,1]


    def read_data(self):
        '''Loads the X and Y data for training, validation or test data set.'''
        datadir = self.datadir
        data_set = self.data_set
        ## Load Y data. 
        Y = self.read_data_Y()

        ## Load X data. 
        if 'XGB' in self.model_name:
            X_train = np.load(f'{datadir}/X_train.npy')
            X_val = np.load(f'{datadir}/X_val.npy')
            X_test = np.load(f'{datadir}/X_test.npy')

            X_train,X_val,X_test = normalize_2D_data_notY(X_train, X_test, X_val)
            return_two_inputs = False

        elif self.model_name == 'naive_week':
            X_train = np.load(f'{datadir}/X_week_train.npy')[:,:,-2:]
            X_val = np.load(f'{datadir}/X_week_val.npy')[:,:,-2:]
            X_test = np.load(f'{datadir}/X_week_test.npy')[:,:,-2:]
            # ^should correctly select soc and is_home, no normalization needed.
            return_two_inputs = False
        else:
            X_week_train = np.load(f'{datadir}/X_week_train.npy')
            X_week_val = np.load(f'{datadir}/X_week_val.npy')
            X_week_test = np.load(f'{datadir}/X_week_test.npy')

            X_yest_train = np.load(f'{datadir}/X_yest_train.npy')
            X_yest_val = np.load(f'{datadir}/X_yest_val.npy')
            X_yest_test = np.load(f'{datadir}/X_yest_test.npy')

            X_week_train, X_week_val, X_week_test, scalers_X = normalize_3D_data_notY(X_week_train, X_week_test, X_week_val, set_scaler='MinMaxScaler')
            X_yest_train, X_yest_val, X_yest_test, scalers_X = normalize_3D_data_notY(X_yest_train, X_yest_test, X_yest_val, set_scaler='MinMaxScaler')
            return_two_inputs = True

        if return_two_inputs:
            X_train = (X_yest_train, X_week_train)
            X_val = (X_yest_val, X_week_val)
            X_test = (X_yest_test, X_week_test)
        
        if data_set == 'train':
            return X_train, Y
        elif data_set == 'val':
            return X_val, Y
        elif data_set == 'test':
            return X_test, Y


    def read_data_Y(self, data_set=None):
        '''Reads the Y data for training, validation or test data.'''
        datadir = self.datadir
        if not data_set:
            data_set = self.data_set
        
        if data_set == 'train':
            Y = np.load(f'{datadir}/Y_train.npy')
        elif data_set == 'val':
            Y = np.load(f'{datadir}/Y_val.npy')
        elif data_set == 'test':
            Y = np.load(f'{datadir}/Y_test.npy')
        return Y


    def compute_predictions(self):
        if 'XGB' in self.model_name:
            X, Y = self.read_data()
            preds_soc, preds_home = self.model.predict(X)
            true = Y
            
            soc_data = {
                'true' : true[:,:H],
                'preds' : preds_soc
            }
            home_data = {
                'true' : true[:,H:],
                'preds' : preds_home
            }

        elif 'naive' in self.model_name:
            X, Y = self.read_data()
            preds_soc, preds_home = self.model.predict(X)
            true = Y

            soc_data = {
                'true' : true[:,:,0],
                'preds' : preds_soc
            }
            home_data = {
                'true' : true[:,:,1],
                'preds' : preds_home
            }

        else:
            X, Y = self.read_data()
            preds_soc, preds_home = self.model.predict(X)
            true = Y

            soc_data = {
                'true' : true[:,:,0],
                'preds' : preds_soc
            }
            home_data = {
                'true' : true[:,:,1],
                'preds' : preds_home
            }            
        return soc_data, home_data
    

    def compute_opt_threshold_and_auc(self):
        '''
        Depreciated function. The models (not naive though) output probability of the EV being at home, 
        and in order to make a guess, a threshold must be assigned. This can be optimized according to 
        risk-aversion by the user. This was an attempt to optimize thresholds for each horizon by looking 
        at the training data. However, this did not prove to generalize well, and so the simple threshold of
        0.5 was used. 
        '''
        if 'naive' in self.model_name:
            return None,None

        true_train = self.home_data['true_train']
        preds_train = self.home_data['preds_train']
        true_val = self.home_data['true_val']
        preds_val = self.home_data['preds_val']

        H = true_train.shape[1]
        stats_dict = {}
        opt_thresholds = []
        for h in range(H):
            fpr, tpr, thresholds = roc_curve(true_train[:,h], preds_train[:,h])
            opt_thresholds.append(thresholds[np.argmax(tpr-fpr)])

            fpr,tpr,_ = roc_curve(true_val[:,h],preds_val[:,h])
            auc = roc_auc_score(true_val[:,h], preds_val[:,h])
            stats_dict[h] = {'fpr':fpr,'tpr':tpr,'auc':auc}

        return opt_thresholds, stats_dict



def plot_rmse(eval, save=False, savedir = None):
    rmses = stats_per_horizon(eval.soc_data['true'],eval.soc_data['preds'],task = 'rmse')

    fig = plt.figure(figsize=pref_figsize)
    plt.plot(rmses)
    plt.xlabel('Horizon')
    plt.ylabel('RMSE')
    if eval.data_set == 'val':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} validation data')
    elif eval.data_set == 'test':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} test data')
    
    if save:
        fig.savefig(f'{savedir}/rmse.png', format='png')


def plot_mae(eval, save=False,savedir = None):
    mae = stats_per_horizon(eval.soc_data['true'],eval.soc_data['preds'],task = 'mae')

    fig = plt.figure(figsize=pref_figsize)
    plt.plot(mae)
    plt.xlabel('Horizon')
    plt.ylabel('MAE')
    if eval.data_set == 'val':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} validation data')
    elif eval.data_set == 'test':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} test data')
    
    if save:
        fig.savefig(f'{savedir}/mae.png', format='png')


def plot_acc_and_f1(eval, save=False, savedir = None):
    f1scores = get_f1score(eval.home_data['true'],eval.home_data['preds'],thresholds=None)
    accuracies = stats_per_horizon(eval.home_data['true'],eval.home_data['preds'],task = 'clf',thresholds=None)

    fig_f1 = plt.figure(figsize=pref_figsize)
    plt.plot(f1scores)
    plt.xlabel('Horizon')
    plt.ylabel('F1 score')
    if eval.data_set == 'val':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} validation data')
    elif eval.data_set == 'test':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} test data')

    fig_acc = plt.figure(figsize=pref_figsize)
    plt.plot(accuracies)
    plt.xlabel('Horizon')
    plt.ylabel('Accuracy')
    if eval.data_set == 'val':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} validation data')
    elif eval.data_set == 'test':
        plt.title(f'{eval.disp_name} on FCR {eval.fcr} test data')

    
    if save:
        fig_f1.savefig(f'{savedir}/f1.png', format='png')
        fig_acc.savefig(f'{savedir}/acc.png', format='png')


def plot_ROC_curve(eval, save=False, savedir = None):
    '''
    Illustrates the tradeoff between False Positive Rate (FPR) and True Positive Rate (TPR)
    depending on choice of threshold. Visualized for the 1st, middle and last predicted time steps.
    This may be used to determine qualified choices of thresholds for the binary classifiers to 
    find a trade-off best suited for the use-case. A perfect predictor would have 
    TPR = 1 and FPR = 0. 
    '''
    check_these = [0,24,48]
    fig= plt.figure(figsize=(6,6))

    for c in check_these:
        fpr, tpr, _ = roc_curve(eval.home_data['true'][:,c], eval.home_data['preds'][:,c])
        auc = roc_auc_score(eval.home_data['true'][:,c], eval.home_data['preds'][:,c])
        auc = round(auc,3)

        plt.plot(fpr, tpr, label=f'Horizon {c+1} AUC: {auc}')
    
    plt.plot(np.linspace(start=0,stop=1),np.linspace(start=0,stop=1),ls='--',c='C0')
    if eval.data_set == 'val':
        plt.title(f'ROC {eval.disp_name} on FCR {eval.fcr} test data')
    elif eval.data_set == 'test':
        plt.title(f'ROC {eval.disp_name} on FCR {eval.fcr} test data')
        
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()

    if save:
        fig.savefig(f'{savedir}/ROC.png', format='png')


def main(model_name):
    ## Load model compute predictions on train and val data,
    # and compute optimal thresholds and FPR, TPR and AUC.

    print(f'####### starting evaluation on {model_name}')
    
    for dataset in DATASET_LIST:
        model_eval = Evaluate(model_name, data_set = dataset, load_data = False)

        ## Save images to model directory
        savedir = f'{model_eval.savedir}/plots/{dataset}'
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        plot_acc_and_f1(model_eval,save=True,savedir=savedir)
        plot_rmse(model_eval,save=True,savedir=savedir)
        plot_mae(model_eval,save=True,savedir=savedir)

        if 'naive' not in model_name: # ROC curve is not applicable for naive baseline.
            plot_ROC_curve(model_eval,save=True,savedir=savedir)


if __name__ == '__main__': 
    for model in MODELS_LIST:
        main(model_name = model)





