import model
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, cpu=False):
    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (25, 1)
        conv_filters2 = (1, Chans)
        pool_size = (45, 1)
        strides = (15, 1)
        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, 20)
        conv_filters2 = (Chans, 1)
        pool_size = (1, 45)
        strides = (1, 15)
        axis = 1

    input_main = Input(input_shape)
    block1 = Conv2D(20, conv_filters,
                    input_shape=input_shape,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(20, conv_filters2, use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)