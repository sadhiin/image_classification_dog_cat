import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation


def CNN_Model():
    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Conv2D(64, (3,3), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3,3), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(256,(3,3), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Step 3 - Flattening
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    return model