#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:48:58 2018

@author: limeng
"""

import numpy as np

from keras.models import load_model
from keras.utils import plot_model
import keras.backend as K

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
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15))
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)

# make a line for the .pdb file
def makeLine(coord, k):
    atom = 'ATOM'
    atom_sn = str(k+1)
    atom_name = 'D1'
    res_name = 'LIG'
    E = 0.1
    x = '{:.3f}'.format(round(coord[0], 3))
    y = '{:.3f}'.format(round(coord[1], 3))
    z = '{:.3f}'.format(round(coord[2], 3))
    EE = '{:.2f}'.format(round(E,2))
    string = atom + ' '*2 + '{:>5}'.format(atom_sn) + ' ' + '{:4}'.format(atom_name) + ' ' \
            + '{:>3}'.format(res_name) + ' '*2 + '   1' + ' '*4 + '{:>8}'.format(x) + '{:>8}'.format(y) + '{:>8}'.format(z) \
            + '{:>6}'.format('1.00') + '{:>6}'.format(EE) + ' '*8 + '\n'
    return string

protein_name = '13pkA'
data_folder = '/home/limeng/Desktop/pocket_similarity/model/voxel/voxelization/voxel_data/'
data_path = data_folder + protein_name + '.npy'

pocket_voxel = np.load(data_path)
X = np.transpose(pocket_voxel, (1,2,3,0))
X = np.expand_dims(X, axis = 0)
res_mdl = load_model('./voxresnet_models/DeepDrug_VoxResNet_ver0.h5', custom_objects={'tversky_loss': tversky_loss,
                                                                                  'weighted_dice_coef':weighted_dice_coef,
                                                                                  'dice_coef_loss':dice_coef_loss})
plot_model(res_mdl, show_shapes=True, to_file='VoxResNet.png')
ligand_voxel = res_mdl.predict(X)
y = np.squeeze(ligand_voxel)

voxel_start = -15
voxel_end = 16
l = []
for a in xrange(32):
    for b in xrange(32):
        for c in xrange(32):
            tempy = y[a,b,c,:]
            at = np.argmax(tempy)
            temp = [a + voxel_start,b + voxel_start,c + voxel_start, at]
            l.append(temp)
            
atom_type_list = ['C.2','C.3','C.ar','F','N.am','N.2','O.co2','N.ar','S.3','O.2','O.3','N.4','P.3','N.pl3']
with open('./13pkA_VoxResNet.pdb','w') as in_strm:
    for k in xrange(len(l)):
        type_code = l[k][3]
#        if type_code != 0 and type_code != 12 and type_code != 13:
        if type_code != 0:
            print(atom_type_list[type_code-1])
            temp_c = l[k][0:3]
            temp_string = makeLine(temp_c, k)
            in_strm.write(temp_string)
        else:
            continue
in_strm.close()
del res_mdl
K.clear_session()