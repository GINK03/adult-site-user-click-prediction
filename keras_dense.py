from keras.layers import Dense
from keras.layers               import Input, Dense, SimpleRNN, GRU, LSTM, CuDNNLSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Flatten
from keras.callbacks            import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers               import merge
from keras.applications.vgg16   import VGG16 
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Concatenate
from keras.layers.core          import Dropout
from keras.layers.merge         import Concatenate as Concat
from keras.layers.noise         import GaussianNoise as GN
from keras.layers.merge         import Dot,Multiply
from keras import backend as K

import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re
import time
import json
import gzip
from sklearn.cross_validation import KFold

npz = np.load('dense.npz')

for file in npz.files:
  print(file)

clks, t_hots = npz['clks'], npz['t_hots']
w_freqs      = npz['w_freqs']

for c in clks:
  print(c)

def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def getModel():
  PERD_SIZE     = 1
  input_tensor1 = Input(shape=(3000,))

  x             = input_tensor1
  x             = Dense(1000, activation='relu')(x)
  x1            = Dropout(0.5)(x)
  
  
  input_tensor2 = Input(shape=(3000,))
  x             = Dense(1000, activation='relu')(x)
  x2            = Dropout(0.5)(x)

  x3            = Concatenate(axis=-1)( [x1, x2] )
  x3            = Dense(1000, activation='relu')(x3)

  prediction    = Dense(PERD_SIZE, activation='sigmoid')(x3)

  model         = Model([input_tensor1, input_tensor2], prediction)
  model.compile( Adam(lr=0.0001), loss='mae')
  return model

from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
from sklearn.cross_validation import KFold
NFOLDS=5
SEED=71
kf = KFold(len(t_hots), n_folds=NFOLDS, shuffle=True, random_state=SEED)

for ki, (train_index, test_index) in enumerate(kf):
  t_hots_tr, t_hots_val = t_hots[train_index], t_hots[test_index]
  w_freqs_tr, w_freqs_val = w_freqs[train_index], w_freqs[test_index]
  clks_tr, clks_val = clks[train_index], clks[test_index]
  model = getModel()
  for epoch in range(50):
    hist = model.fit([t_hots_tr, w_freqs_tr], clks_tr, validation_data=([t_hots_val, w_freqs_val], clks_val), batch_size=2*512)
    nextlr = 0.98 * K.get_value(model.optimizer.lr)
    loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    print('kf', ki, 'epoch', epoch, 'next lr', nextlr)
    K.set_value(model.optimizer.lr, nextlr)
    
    model.save_weights(f'models/{ki:04d}_{epoch:04d}_{val_loss:0.08f}_{loss:0.08f}.h5')
