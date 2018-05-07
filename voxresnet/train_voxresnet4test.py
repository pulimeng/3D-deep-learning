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

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=14)[...,1:])
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
with open('../voxel/voxelization/atp_names.lst') as atp_in:
    for line in atp_in.readlines():
        temp = line.replace(' ','').replace('\n','')
        atps.append(temp)
# conver data into a single matrix
voxel_folder = '../voxel/voxelization/voxel_data/'
voxel = np.zeros(shape = (10, 32, 32, 32, 14),
    dtype = np.float64)
label_folder = './ligand-voxel/'
label = np.zeros(shape = (10, 32, 32, 32, 1),
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
    if cnt == 10:
        break
print 'compuation time for data conversion is ' + str(time.time()-ss)

batch_size = 2
epoch = 2
X_train, X_test, y_train, y_test = train_test_split(voxel, label)

model = VoxResNetBuilder.build_voxresnet((32, 32, 32, 14), 14)
print_summary(model, line_length=120, positions=None, print_fn=None)
sgd = optimizers.SGD(lr=0.0001, momentum=0.99)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = sgd)
tfCallBack = callbacks.TensorBoard(log_dir='./graph', histogram_freq = 0, batch_size=16, write_graph=True, 
					write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model.fit_generator(generate_batch_data_random(X_train, y_train, batch_size),                                                      
    steps_per_epoch=len(y_train)//batch_size*batch_size,
    epochs=epoch,
    verbose=2,
    callbacks=[tfCallBack])
#model.fit(X_train, y_train, batch_size=10, epochs = 1, shuffle = True, 
#          callbacks = [tfCallBack], verbose = 2)
scores = model.evaluate(X_test, y_test,verbose = 1)