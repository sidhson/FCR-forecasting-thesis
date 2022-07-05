'''
The naive baseline model used in the thesis. Predicts the exact same behaviour as the week prior.
No parameters or training for this model, only important to feed prediction with correctly indexed 
X-data. 
'''

import numpy as np


class NaiveWeek:
    def __init__(self):
        self.model_name = 'naive_week'


    def predict(self,X):
        '''
        give the model X_week sequence, and select indices 0 and 1 for soc and is_home. 
        thus when the naive model gets the x-data, it has only 2 columns in each sequence, 
        0 for soc and 1 for is_home. When the model is called it is important that soc and 
        is_home are ordered in this way. 
        '''
        reg_preds, bin_preds = X[:,:,0],X[:,:,1]
        return reg_preds, bin_preds
