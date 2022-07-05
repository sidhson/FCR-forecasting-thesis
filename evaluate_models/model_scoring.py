import numpy as np
from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,mean_absolute_error

def stats_per_horizon(true,preds,task = 'rmse',thresholds = None):
    '''
    Assumes inputs to be of shape (N,H). Can evaluate regression or classification forecasters. 
    Regression: MAE or RMSE
    Classification: Accuracy
    '''
    scores = []
    H = true.shape[1]
    if task == 'rmse':
        for h in range(H):
            scores.append(mean_squared_error(true[:,h],preds[:,h],squared = False))

    elif task == 'mae':
        for h in range(H):
            scores.append(mean_absolute_error(true[:,h],preds[:,h]))

    elif task == 'clf':
            # if none, 0.5 as default, AND if predictions are already binary,
            # then it will still make sense. 
            # NOTE: for future, if super-optimized models (thresholds)
            # then threshold-list for different horizons potentially. 
        if thresholds is None:
            thresholds = 0.5*np.ones(shape=(H,))
        for h in range(H):
            rounded = []
            for x in preds[:,h]:
                rounded.append(int(x >= thresholds[h]))
            scores.append(accuracy_score(true[:,h],rounded)) # preds[:,h]

    return scores


def get_f1score(true,preds,thresholds=None):
    H = true.shape[1]
    if thresholds is None:
            thresholds = 0.5*np.ones(shape=(H,))
    scores = []

    for h in range(H):
        rounded = []
        for x in preds[:,h]:
            rounded.append(int(x>=thresholds[h]))
        scores.append(f1_score(true[:,h],rounded))
    return scores


def distr_per_horizon(true,preds,task = 'rmse',thresholds = None, percentile=5):
    '''
    Assumes inputs to be of shape (N,H). Can evaluate regression or classification 
    forecasters performance with a assumed normal distribution of the errors over horizons. 
    Percentile is per default 5. 
    
    Regression: MAE, RMSE
    Classification: Accuracy, F1-score

    Returns a vector of of horizons with (2.5 percentile, mean, 97.5 percentile) per step.
    '''
    H = true.shape[1]
    scores_distr = np.empty(shape=(H,3))
    if task == 'rmse':
        for h in range(H):
            pass
            squared_err = (true[:,h] - preds[:,h])**2
            lb_err = np.sqrt(np.percentile(squared_err, percentile/2))
            ub_err = np.sqrt(np.percentile(squared_err, 100-percentile/2))
            mean_err = np.sqrt(np.mean(squared_err))
            scores_distr[h,:] = [lb_err, mean_err, ub_err]

    elif task == 'mae':
        for h in range(H):
            abs_err = np.abs(true[:,h] - preds[:,h])
            lb_err = np.percentile(abs_err, percentile/2)
            ub_err = np.percentile(abs_err, 100-percentile/2)
            mean_err = np.mean(abs_err)
            scores_distr[h,:] = [lb_err, mean_err, ub_err]

    elif task == 'clf':
            # if none, 0.5 as default, AND if predictions are already binary,
            # then it will still make sense. 
            # NOTE: for future, if super-optimized models (thresholds)
            # then threshold-list for different horizons potentially. 
        if thresholds is None:
            thresholds = 0.5*np.ones(shape=(H,))
        for h in range(H):
            rounded = []
            for x in preds[:,h]:
                rounded.append(int(x >= thresholds[h]))
            
            scores_distr[h,:] = [0, accuracy_score(true[:,h],rounded), 0]

    return scores_distr