import json 
import numpy as np 
import os
import sys
import xgboost as xgb

cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from preprocessing.gen_cross_data import get_X_columns

H = 49
RANDOM_SEED = 1

REG_CONFIG_BEST = {
    'only_day': True,
    'n_estimators' : 400,
    'colsample_bytree' : 0.75,
    'subsample' : 0.5,
    'max_depth' : 6,
    'learning_rate':0.1,
    'booster':'dart',
    'rate_drop':0.1,
    'gamma':0,
    'recursive':False,
    'tree_method':'gpu_hist'
}
BIN_CONFIG_BEST = {
    'only_day' : False,
    'n_estimators':200,
    'colsample_bytree':0.75,
    'subsample':0.5,
    'max_depth':6,
    'learning_rate':0.3,
    'booster':'dart',
    'rate_drop':0.1,
    'gamma':0.0001,
    'recursive':'False',
    'tree_method':'gpu_hist' 
}

'''
The lines below give lists of indexes in all of the columns generated falling into 
the yesterday or last week sequence. Functionality to exclude features based on attribute rather
than time can be added. 
'''
all_cols = get_X_columns()
indexes_only_day = []
for i,feat in enumerate(all_cols):
    sm = 0
    yest = False
    spl = feat.split('-')
    if len(spl) == 1:
        yest = True
    elif int(spl[-1]) < 50:
        yest = True
    if yest:
        indexes_only_day.append(i)
indexes_week = [i for i in range(len(all_cols))]


def load_model(read_dir,verbose = True):
    '''
    As the configurations are set before the model object is created, this is the 
    function that should be called to load a saved model. 
    The directory should contain configurations for the regressors and binary classifiers, 
    as well as the weights of the individual predictors. Perhaps best understood by first examining 
    the save_self-function in the XGBForecaster class.
    '''
    with open(f'{read_dir}/reg_config.json') as f:
        reg_config = json.load(f)
    with open(f'{read_dir}/bin_config.json') as f:
        bin_config = json.load(f)
    model = XGBForecaster(reg_config,bin_config)
    model._load_predictors(read_dir,verbose=verbose)
    return model


class XGBForecaster:
    '''
    The XGBForecaster class is a structured collection of individual single-output 
    XGBoost models, 49 regressors and 49 binary classifiers. The configurations of said
    sub-models are set separately for the regressors and the classifiers. 
    '''
    def __init__(self,reg_config = REG_CONFIG_BEST,bin_config=BIN_CONFIG_BEST):

        self.reg_config = reg_config
        self.bin_config = bin_config
        regs = []
        bins = []
        if self.reg_config['only_day']:
            self.reg_colindexes = indexes_only_day
        else:
            self.reg_colindexes = indexes_week
        if self.bin_config['only_day']:
            self.bin_colindexes = indexes_only_day
        else:
            self.bin_colindexes = indexes_week

        for _ in range(H):
            regs.append(
                xgb.XGBRegressor(
                    n_estimators = self.reg_config['n_estimators'],
                    colsample_bytree = self.reg_config['colsample_bytree'],
                    subsample = self.reg_config['subsample'],
                    gamma = self.reg_config['gamma'],
                    max_depth = self.reg_config['max_depth'], 
                    tree_method = self.reg_config['tree_method'], 
                    learning_rate = self.reg_config['learning_rate'],
                    booster = self.reg_config['booster'],
                    rate_drop = self.reg_config['rate_drop'],
                    random_state = RANDOM_SEED
                )
            ) 
            bins.append(
                xgb.XGBClassifier(
                    n_estimators = self.bin_config['n_estimators'],
                    colsample_bytree = self.bin_config['colsample_bytree'],
                    subsample = self.bin_config['subsample'],
                    gamma = self.bin_config['gamma'],
                    max_depth = self.bin_config['max_depth'],
                    tree_method = self.bin_config['tree_method'],
                    learning_rate = self.bin_config['learning_rate'],
                    booster = self.bin_config['booster'], 
                    rate_drop = self.bin_config['rate_drop'],
                    objective = 'binary:logistic',
                    use_label_encoder = False, 
                    random_state = RANDOM_SEED
                )
            )
        self.regressors = regs
        self.classifiers = bins
    
    def fit(self,X,Y,verbose = True):
        '''
        All models in the collection are fitted to the data. 
        Regressors and classifiers may have varying input data, the information on this 
        is carried in their respective colindexes.
        When constructing the object, classifiers and regressors may be set to recursive. In this case, 
        each predictor is fed with the cumulative predictions of the previous predictors. 
        Thus far, this has not shown to be advantageous. 
        Note that the input data X is partitioned between the classifiers and regressors according
        to their respective configurations. 
        '''
        regs = Y[:,:H]
        bins = Y[:,H:]
        
        X_reg = X[:,self.reg_colindexes]
        X_bin = X[:,self.bin_colindexes]

        for h in range(H):
            if verbose:
                print(f'Training both tasks for horizon {h+1}')
            if h > 0 and self.reg_config['recursive']:
                reg_hat = self.regressors[h-1].predict(X_reg)
                X_reg = np.c_[X_reg,reg_hat]
            if h > 0 and self.bin_config['recursive']:
                bin_hat = self.classifiers[h-1].predict(X_bin)
                X_bin = np.c_[X_bin,bin_hat]

            self.regressors[h].fit(X_reg,regs[:,h])
            self.classifiers[h].fit(X_bin,bins[:,h])
    
    def predict(self,X,verbose = False):
        ''''
        Makes predictions in aggregation (X input is assumed to cover several observations). 
        Works analogously as the fit-function. 
        '''
        n = X.shape[0]
        reg_preds = np.empty(shape = (n,H))
        bin_preds = np.empty(shape = (n,H)) 

        X_reg = X[:,self.reg_colindexes]
        X_bin = X[:,self.bin_colindexes]

        for h in range(H):
            if verbose:
                print(f'Predicting both tasks for horizon {h+1}')
            if h > 0 and self.reg_config['recursive']:
                reg_hat = reg_preds[:,h-1]
                X_reg = np.c_[X_reg,reg_hat]
            if h > 0 and self.bin_config['recursive']:
                bin_hat = bin_preds[:,h-1]
                X_bin = np.c_[X_bin,bin_hat]
            reg_preds[:,h] = self.regressors[h].predict(X_reg)
            double_probs = self.classifiers[h].predict_proba(X_bin)
            bin_preds[:,h] = double_probs[:,1]
        return reg_preds,bin_preds

    def _load_predictors(self,indir,verbose=True):
        '''
        This method is called after the configurations have been read. The standard 
        XGBoost models can load weights from json, which are the ones created when 
        the model is saved. 
        '''
        for i in range(H):
            if verbose:
                print(f'Loading both models for horizon {i+1} out of {H}...')
            self.regressors[i].load_model(f'{indir}/reg_{i}.json')
            self.classifiers[i].load_model(f'{indir}/bin_{i}.json')

    def _save_predictors(self,outdir,verbose = True):
        '''
        An XGBoost model object has the function save_model, which saves the model weights 
        to a json-file. This is done for each of the predictors in the structured collection, 
        and they are named according to reg for regressors and bin for the binary classifiers. 
        The indexing is from 0. 
        '''
        for i in range(H):
            if verbose:
                print(f'Saving both models for horizon {i+1} out of {H}...')
            self.regressors[i].save_model(f'{outdir}/reg_{i}.json')
            self.classifiers[i].save_model(f'{outdir}/bin_{i}.json')

    def save_self(self,outdir,verbose = True):
        '''
        Save the trained model for later use without need for re-training. 
        Model is saved according to the following: 
            - Create a designated directory for the model (preferably empty)
            - This method begins by dumping the configurations for regressors and classifiers
              to json files. 
            - It then calls on _save_predictors, which in turn saves the weights of each predictor.
        '''
        with open(f'{outdir}/reg_config.json','w') as f:
            json.dump(self.reg_config,f)
        with open(f'{outdir}/bin_config.json','w') as f:
            json.dump(self.bin_config,f)

        self._save_predictors(outdir,verbose)