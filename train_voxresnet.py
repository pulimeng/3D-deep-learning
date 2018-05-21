#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Train the VoxResNet

@author: limeng
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'

import itertools
import numpy as np
from scipy import interp

#import h5py
import time

import keras.backend as K
from keras.utils import np_utils
from keras import optimizers
from keras import callbacks
from keras.utils import print_summary
from voxresnet import VoxResNetBuilder

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

seed = 12306
np.random.seed(seed)

def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = np.random.randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
        
def tversky_loss(y_true, y_pred):
    alpha = 0.9
    beta  = 0.1
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def weighted_dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_0 = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15)[...,0])
    y_pred_0 = K.flatten(y_pred[...,0])
    intersect_0 = K.sum(y_true_0 * y_pred_0, axis=-1)
    denom_0 = K.sum(y_true_0 + y_pred_0, axis=-1)
    y_true_1 = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15)[...,1:])
    y_pred_1 = K.flatten(y_pred[...,1:])
    intersect_1 = K.sum(y_true_1 * y_pred_1, axis=-1)*10000
    denom_1 = K.sum(y_true_1 + y_pred_1, axis=-1)*10000
    intersect = intersect_0 + intersect_1
    denom = denom_0 + denom_1
    return K.mean((2. * intersect / (denom + smooth)))

def weighted_dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - weighted_dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)


# load the ATP data
atps = []
with open('./atp_names.lst') as atp_in:
    for line in atp_in.readlines():
        temp = line.replace(' ','').replace('\n','')
        atps.append(temp)
# conver data into a single matrix
voxel_folder = './voxel_data/'
voxel = np.zeros(shape = (1553, 32, 32, 32, 14),
    dtype = np.float64)
label_folder = './ligand-voxel/'
label = np.zeros(shape = (1553, 32, 32, 32, 1),
    dtype = np.int64)
cnt = 0
ss = time.time()
print '...Loading the data'
for atp in atps:
    v_path = voxel_folder + atp + '.npy'
    l_path = label_folder + atp + '.npy'
    v = np.load(v_path)
    l = np.load(l_path)
    v = np.transpose(v, (1,2,3,0))
    l = np.transpose(l, (1,2,3,0))
    voxel[cnt,:] = v
    label[cnt,:] = l
    cnt += 1
print 'compuation time for data conversion is ' + str(time.time()-ss)

batch_size = 32
epoch = 30
X_train, X_test, y_train, y_test = train_test_split(voxel, label)

model = VoxResNetBuilder.build_voxresnet((32, 32, 32, 14), 15)
print_summary(model, line_length=120, positions=None, print_fn=None)

opt = optimizers.Adam(lr=0.001)
model.compile(loss = dice_coef_loss, optimizer = opt, metrics = ['sparse_categorical_accuracy', tversky_loss])
tfCallBack = callbacks.TensorBoard(log_dir='./graph', histogram_freq = 0, batch_size=batch_size, write_graph=True, 
					write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model.fit_generator(generate_batch_data_random(X_train, y_train, batch_size),                                                      
    steps_per_epoch=len(y_train)//batch_size,
    epochs=epoch,
    verbose=2,
    callbacks=[tfCallBack])
#model.fit(X_train, y_train, batch_size=10, epochs = 1, shuffle = True, 
#          callbacks = [tfCallBack], verbose = 2)
scores = model.evaluate(X_test, y_test,verbose = 1)
print(scores)
model.save('saved_models/DeepDrug_VoxResNet_ver0.h5')