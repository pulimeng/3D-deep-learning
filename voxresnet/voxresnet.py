from __future__ import (
    division,
    print_function,
    unicode_literals
)
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Merge,
    Reshape,
    Lambda
)
from keras.layers.convolutional import (
    Conv3D,
    Conv3DTranspose
)
from keras.layers.merge import Concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = input._keras_shape[DIM1_AXIS] \
        // residual._keras_shape[DIM1_AXIS]
    stride_dim2 = input._keras_shape[DIM2_AXIS] \
        // residual._keras_shape[DIM2_AXIS]
    stride_dim3 = input._keras_shape[DIM3_AXIS] \
        // residual._keras_shape[DIM3_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, kernel_regularizer, repetitions, 
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
#            if i == 0 and not is_first_layer: #TODO change the stride voxres has stride 1
#                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False):
    """Bottleneck 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
#        if is_first_block_of_first_layer: #TODO change the first conv block setup
#            # don't repeat bn->relu since we just did bn->relu->maxpool
#            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
#                              strides=strides, padding="same",
#                              kernel_initializer="he_normal",
#                              kernel_regularizer=kernel_regularizer
#                              )(input)
#        else:
        conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                   strides=strides,
                                   kernel_regularizer=kernel_regularizer
                                   )(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def custom_activation(x):
    return K.cast(K.expand_dims(K.argmax(x, axis = CHANNEL_AXIS),axis = CHANNEL_AXIS),'int64')

class VoxResNetBuilder(object):
    """VoxResNet."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, reg_factor):
        """Instantiate a VoxResNet keras model.
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        get_custom_objects().update({'argmax_activation': Activation(custom_activation)})
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        v_input = Input(shape=input_shape)
        # conv1
        conv1a = _conv_bn_relu3D(filters=32, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(v_input)
        conv1b = _conv_bn_relu3D(filters=32, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                kernel_regularizer=l2(reg_factor)
                                )(conv1a)
        # C1
        C1d = Conv3DTranspose(filters = 32, kernel_size = (3,3,3),
                             strides=(1, 1, 1),
                             kernel_regularizer=l2(reg_factor),
                             padding = 'same'
                             )(conv1b)
        C1 = Conv3D(filters = 32, kernel_size = (1,1,1),
                    strides = (1,1,1),  
                    padding = 'same',
                    kernel_regularizer=l2(reg_factor)
                    )(C1d)
        
        conv1c = Conv3D(filters=64, kernel_size=(3, 3, 3),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor),
                                padding = 'same'
                                )(conv1b)

        # voxres block2-3
        filters = 64
        block23 = _residual_block3d(block_fn, filters=filters,
                                  kernel_regularizer=l2(reg_factor),
                                  repetitions=2, is_first_layer=False
                                  )(conv1c)
        # C2
        C2d = Conv3DTranspose(filters = filters, kernel_size = (3,3,3),
                             strides=(2, 2, 2),
                             kernel_regularizer=l2(reg_factor),
                             padding = 'same')(block23)
        
        C2 = Conv3D(filters = filters, kernel_size = (1,1,1),
                    strides = (1,1,1),
                    padding = 'same',
                    kernel_regularizer=l2(reg_factor)
                    )(C2d)

        # concatenation and segmentation
        C = Concatenate(axis = -1)([C1, C2])
        C = Conv3D(filters = num_outputs, kernel_size = (1,1,1),
            strides = (1,1,1),
            padding = 'same',
            kernel_regularizer=l2(reg_factor)
            )(C)
#        F = Reshape((input_shape[0]*input_shape[1]*input_shape[2]*num_outputs,))(C)
        C = Activation('softmax')(C)
#        C = Lambda(custom_activation, output_shape = (input_shape[0],input_shape[1],input_shape[2],1))(C)
        
        model = Model(inputs=v_input, outputs=C)
        return model
    
    @staticmethod
    def build_voxresnet(input_shape, num_outputs, reg_factor=1e-4):
        return VoxResNetBuilder.build(input_shape, num_outputs, basic_block, 
                                      reg_factor=reg_factor)