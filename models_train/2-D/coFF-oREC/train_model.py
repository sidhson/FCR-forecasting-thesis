import numpy as np
import os
import sys
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
import yaml
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Reshape, RepeatVector, TimeDistributed, Bidirectional, Input, Concatenate, Flatten
from tensorflow.keras import Model
from tensorflow.keras.regularizers import L2,L1


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# Set random seed for reproduceability. Not ensured for GPU though. 
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


cdir = os.getcwd()
root = os.path.join(cdir.split('Emulate')[0], 'Emulate-thesis')
sys.path.append(root)

from preprocessing.feature_engineering import normalize_3D_data_notY
from preprocessing.gen_seq_data import FEATURES as colnames


DATE = '2022-05-16'
FCR = '2-D'

ALL_FEATURES = [i for i in range(15)]
FEATURES = ALL_FEATURES
DESELECT_FEATURES = []
FEATURES = [i for i in range(len(colnames)) if colnames[i] not in DESELECT_FEATURES]

datadir = f'{root}/data/train_{FCR}/{DATE}/seq_data'

def slice_features(X,features=ALL_FEATURES):
    return X[:,:,features]



### LOAD DATA ### 
X_week_train = slice_features(np.load(f'{datadir}/X_week_train.npy'),FEATURES)
X_week_val = slice_features(np.load(f'{datadir}/X_week_val.npy'),FEATURES)
X_week_test = slice_features(np.load(f'{datadir}/X_week_test.npy'),FEATURES)

X_yest_train = slice_features(np.load(f'{datadir}/X_yest_train.npy'),FEATURES)
X_yest_val = slice_features(np.load(f'{datadir}/X_yest_val.npy'),FEATURES)
X_yest_test = slice_features(np.load(f'{datadir}/X_yest_test.npy'),FEATURES)

Y_train = np.load(f'{datadir}/Y_train.npy')
Y_val = np.load(f'{datadir}/Y_val.npy')
Y_test = np.load(f'{datadir}/Y_test.npy')

X_week_train, X_week_val, X_week_test, scalers_X = normalize_3D_data_notY(X_week_train, X_week_test, X_week_val, set_scaler='MinMaxScaler')
X_yest_train, X_yest_val, X_yest_test, scalers_X = normalize_3D_data_notY(X_yest_train, X_yest_test, X_yest_val, set_scaler='MinMaxScaler')
# Time encoded sinus-consinus waves needs not be normalized. 

### CONFIG INPUT/OUTPUT SHAPES ###
input_shape = X_yest_train.shape[1:3]
output_shape = Y_train.shape[1:3]


def model_specification(config, input_shape, output_shape) -> Model:

    if config['reg'] == 'l1':
        reg = L1(config['reg_coeff'])
    elif config['reg'] == 'l2':
        reg = L2(config['reg_coeff'])
    elif config['reg'] == 'none':
        reg = None

    # Define layers. 
    input_day = Input(shape = input_shape, name='input_day')
    input_week = Input(shape = input_shape, name='input_week')
    
    lstm_day = LSTM(
            config['lstm_units'], 
            return_sequences=False, 
            kernel_regularizer=reg, 
            dropout=config['lstm_dropout_rate'])
    lstm_week = LSTM(
        config['lstm_units'], 
        return_sequences=False, 
        kernel_regularizer=reg, 
        dropout=config['lstm_dropout_rate'])

    if config['bidirectional']:
        lstm_day = Bidirectional(lstm_day)
        lstm_week = Bidirectional(lstm_week)

    x_day = lstm_day(input_day)
    x_week = lstm_week(input_week)
    

    c = Concatenate()([x_day,x_week])
    c = Dropout(config['dropout_rate'])(c)
    x = Dense(output_shape[0]*config['dense_multiple'], activation='relu', kernel_regularizer=reg)(c)
    x = Dropout(config['dropout_rate'])(x)
    x = Reshape((output_shape[0],config['dense_multiple']))(x)

    lstm_soc = LSTM(units=config['lstm_units'],kernel_regularizer=reg,return_sequences=True)
    lstm_is_home = LSTM(units=config['lstm_units'],kernel_regularizer=reg,return_sequences=True)
    
    x_soc = lstm_soc(x)
    x_is_home = lstm_is_home(x)

    
    y_soc = TimeDistributed(Dense(1,activation = config['soc_activation_out']))(x_soc)
    y_soc = Flatten(name='soc_output')(y_soc)
    y_is_home = TimeDistributed(Dense(1,activation='sigmoid'))(x_is_home)
    y_is_home = Flatten(name='is_home_output')(y_is_home)    
    
    model = Model([input_day,input_week], [y_soc,y_is_home]) # For joint modeling, set list of outputs.
    return model

def train(model_name=None):
    
    ### Single run or Sweep. 
    if model_name:
        ### INIT WandB ### 
        wandb.init(project=f"{model_name}-project-{FCR}", entity="g_and_f")
        fn = f'{root}/models_train/{FCR}/{model_name}/best_config.yaml'
        with open(fn) as fh:
            config_file = yaml.load(fh, Loader=yaml.FullLoader)
            wandb.config = config_file['parameters']
        config = wandb.config
    else:
        ### INIT Sweep ###
        print('### Trying to initate Sweep ###')
        wandb.init()
        config = wandb.config
        print('### Sweep initiated successfully ###')
    print('### Config file:', config)


    ### INIT MODEL AND COMPILE ###
    m = model_specification( 
        config, 
        input_shape=input_shape, 
        output_shape=output_shape
    )

    if config['optimizer'] == 'sgd':
        optimizer = tf.optimizers.SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = tf.optimizers.RMSprop(learning_rate = config['learning_rate'])
    
    m.compile(
        loss = {
            'soc_output':'mse', 
            'is_home_output':'binary_crossentropy'
        }, 
        metrics={
            'soc_output': metrics.RootMeanSquaredError(),
            'is_home_output':metrics.BinaryAccuracy()
        },
        loss_weights = {
            'soc_output':10,
            'is_home_output':1
        },
        optimizer=optimizer
    )

    callbacks = [
        WandbCallback(monitor='val_loss', mode='min', log_weights=False),
        EarlyStopping(monitor='val_loss', mode='min', min_delta=config['min_delta'], patience=config['patience'])
    ]
    print(m.summary())

    
    ### FIT MODEL ###
    print('### Training about to start ###')

    m.fit(
        [X_yest_train,X_week_train],
        [Y_train[:,:,0], Y_train[:,:,1]], # Multi-head output reqires data as list.
        epochs=config['epochs'], 
        batch_size=config['batch_size'], 
        validation_data=([X_yest_val,X_week_val], [Y_val[:,:,0], Y_val[:,:,1]]), # Multi-head output
        verbose=2,
        callbacks=callbacks
    ) # Fit multiple horizon predictor

    print('### Training successful ###')
    print(m.summary())
    wandb.finish()


if __name__ == '__main__':
    MODEL = 'coFF-oREC'
    train(MODEL)