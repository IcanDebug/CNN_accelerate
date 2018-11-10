from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Dense
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.vis_utils import plot_model

from keras import backend as K


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)



def bottleneck(inputs, filters, kernel, t, s):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    return x


def _inverted_dense_block(inputs, filters, kernel, t, strides, n):
    
    if strides == 1:  #all for dense connect
         last_x = inputs
         last_all_layer = [inputs]
         for i in range(0, n):
             x = bottleneck(last_x, filters, kernel, t, 1)
             x = add(last_all_layer.append(x))
             last_x = x
    else:     #the first layer isnot dense connect
        x = bottleneck(inputs, filters, kernel, t, strides)
        last_x = x
        last_all_layer = [x]
         
        for i in range(1, n): #last for dense
            x = bottleneck(last_x, filters, kernel, t, 1)
            last_all_layer.append(x)
            x = add(last_all_layer)
            last_x = x
    return x

def DenseDWNet(input_shape, k):
    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 8, (3, 3), strides=(2, 2))
    x = _inverted_dense_block(x, 16, (3, 3), t=6, strides=2, n=3)
    x = _inverted_dense_block(x, 32, (3, 3), t=6, strides=2, n=4)
    x = _inverted_dense_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_dense_block(x, 128, (3, 3), t=6, strides=2, n=4)
    x = _conv_block(x, 256, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 256))(x)
    x = Dropout(0.3, name='Dropout')(x)
    #x = Conv2D(k, (1, 1), padding='same')(x)
    x = Dense(1024, activation='relu', name='Dense1')(x)

    #x = Activation('softmax', name='softmax')(x)
    #output = Reshape((k,))(x)
    x = Dense(2, activation='sigmoid', name='output')(x)
    output = Reshape((k,))(x)
    model = Model(inputs, output)
    plot_model(model, to_file='images/DenseDWNet.png', show_shapes=True)

    return model





#if __name__ == '__main__':
#    model = MobileNetv2((224, 224, 3), 2)
