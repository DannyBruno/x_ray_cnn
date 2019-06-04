'''

Keras CNN. Uses transfer-learning from imagenet.

'''
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import BatchNormalization

import h5py


def imgnet_weights(model):
    '''
    Grab weights for first few layers of imgnet.
    '''

    # Grab weights from file.
    f = h5py.File('vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

    # Select layers and set weights.
    w, b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
    model.layers[1].set_weights = [w, b]

    w, b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
    model.layers[2].set_weights = [w, b]

    w, b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
    model.layers[4].set_weights = [w, b]

    w, b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
    model.layers[5].set_weights = [w, b]

    f.close()


def build_model():
    input_img = Input(shape=(224, 224, 3), name='ImageInput')

    # 2 imagenet layers.
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)

    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)

    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, output=x)
    imgnet_weights(model)
    return model







