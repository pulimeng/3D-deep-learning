#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:15:20 2018

@author: limeng
"""
from keras.models import Model
from keras.layers import (
    Input,
    Concatenate,
    MaxPooling3D,
)
from keras.layers.convolutional import (
    Conv3D,
    Conv3DTranspose
)
from keras.regularizers import l2

IMAGE_ORDERING = 'channels_last'

class voxUnet(object):
    '''
    The fully connected network structure for voxel segmentation.
    # Input: input_shape -- the shape of the input voxel tensor. "channels_last" order.
     n_classes -- number of classes involved in the segmentation task.
    # Output: a model takes a 5D tensor and output the same 5D tensor that each 
      voxel is labelled as one class.
    '''
    @staticmethod
    def build(input_shape, n_classes, reg_factor):
        # Input
        vox_input = Input(shape=(input_shape))
        
        # Contracting 1
        c1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='contract1_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(vox_input)
        c1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='contract1_conv2', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c1)
        c1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='contract1_pool', data_format=IMAGE_ORDERING)(c1)
        
        # Contracting 2
        c2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='contract2_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c1)
        c2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='contract2_conv2', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c2)
        c2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='contract2_pool', data_format=IMAGE_ORDERING)(c2)
    
        # Contracting 3
        c3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='contract3_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c2)
        c3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='contract3_conv2', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c3)
        c3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='contract3_pool', data_format=IMAGE_ORDERING)(c3)
        
        # Contracting 4
        c4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='contract4_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c3)
        c4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='contract4_conv2', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c4)
        c4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='contract4_pool', data_format=IMAGE_ORDERING)(c4)
        
        # Contracting 5
        c5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='contract5_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(c4)
        c5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='contract5_conv2', data_format=IMAGE_ORDERING,
                   kernel_regularizer=l2(reg_factor))(c5)
        c5 = MaxPooling3D((2, 2, 2), strides=None, name='contract5_pool', data_format=IMAGE_ORDERING )(c5)
        
        # Expanding 1
        e1 = Conv3DTranspose(filters = 256, kernel_size = (3,3,3),
                                 strides=(2, 2, 2), name = 'expand1_upsample',
                                 kernel_regularizer=l2(reg_factor),
                                 padding = 'same')(c5)
        e1 = Concatenate(axis = -1, name = 'expand1_concat')([e1, c4])
        e1 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='expand1_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(e1)
        e1 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='expand1_conv2', data_format=IMAGE_ORDERING,
                   kernel_regularizer=l2(reg_factor))(e1)
        
        # Expanding 2
        e2 = Conv3DTranspose(filters = 128, kernel_size = (3,3,3),
                                 strides=(2, 2, 2), name = 'expand2_upsample',
                                 kernel_regularizer=l2(reg_factor),
                                 padding = 'same')(e1)
        e2 = Concatenate(axis = -1, name = 'expand2_concat')([e2, c3])
        e2 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='expand2_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(e2)
        e2 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='expand2_conv2', data_format=IMAGE_ORDERING,
                   kernel_regularizer=l2(reg_factor))(e2)
        
        # Expanding 3
        e3 = Conv3DTranspose(filters = 64, kernel_size = (3,3,3),
                                 strides=(2, 2, 2), name = 'expand3_upsample',
                                 kernel_regularizer=l2(reg_factor),
                                 padding = 'same')(e2)
        e3 = Concatenate(axis = -1, name = 'expand3_concat')([e3, c2])
        e3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='expand3_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(e3)
        e3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='expand3_conv2', data_format=IMAGE_ORDERING,
                   kernel_regularizer=l2(reg_factor))(e3)
        
        # Expanding 4
        e4 = Conv3DTranspose(filters = 32, kernel_size = (3,3,3),
                                 strides=(2, 2, 2), name = 'expand4_upsample',
                                 kernel_regularizer=l2(reg_factor),
                                 padding = 'same')(e3)
        e4 = Concatenate(axis = -1, name = 'expand4_concat')([e4, c1])
        e4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='expand4_conv1', data_format=IMAGE_ORDERING,
                       kernel_regularizer=l2(reg_factor))(e4)
        e4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='expand4_conv2', data_format=IMAGE_ORDERING,
                   kernel_regularizer=l2(reg_factor))(e4)
        
        # Expanding 4
        e5 = Conv3DTranspose(filters = 32, kernel_size = (3,3,3),
                                 strides=(2, 2, 2), name = 'expand5_upsample',
                                 kernel_regularizer=l2(reg_factor),
                                 padding = 'same')(e4)
        
        # Final Convolution
        f = Conv3D(n_classes, (1, 1, 1), activation='softmax', padding='same', name='final_conv', data_format=IMAGE_ORDERING,
                   kernel_regularizer=l2(reg_factor))(e5)
        model = Model(inputs = vox_input, outputs = f)
        
        return model