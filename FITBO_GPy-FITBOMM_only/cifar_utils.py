#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:36:29 2019

@author: jian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:06:52 2019

@author: jian
"""


'''
#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from tensorflow import set_random_seed

num_classes = 10
epochs = 20
steps_per_epoch = 500
data_augmentation = True
num_predictions = 20

initial_size = 2


params_simple = {"batch_size": 32,
          "l1_dropout": 0.25,
          "l2_dropout": 0.25,
          "l3_dropout": 0.5,
          "rms_l_rate": 0.0001,
          "fc_units": 512,
          }

params_complex = {"batch_size": 32,
          "l1_dropout": 0.25,
          "l2_dropout": 0.25,
          "l3_dropout": 0.5,
          "rms_l_rate": 0.0001,
          "l1_conv_filter": 32,
          "l2_conv_filter": 32,
          "l3_conv_filter": 64,
          "l4_conv_filter": 64,
          "dense_units": 512,
          "activation1": "relu",
          "activation2": "relu",
          "activation3": "relu",
          "activation4": "relu",
          }

def cifar_cnn_gpyopt(x):

    # Unwrapping X into parameters
    x = np.ravel(x)

    batch_size = int(x[0])
    fc_units = int(x[1])
    l1_dropout = x[2]
    l2_dropout = x[3]
    l3_dropout = x[4]
    rms_l_rate = x[5]

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', # l1_conv_filter
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(l1_dropout))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(l2_dropout))

    model.add(Flatten())
    model.add(Dense(fc_units))
    model.add(Activation('relu'))
    model.add(Dropout(l3_dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=rms_l_rate, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch = steps_per_epoch,
                            validation_data=(x_test, y_test),
                            workers=4)
        scores = model.evaluate(x_test, y_test, verbose=0)
        test_accuracy = scores[1]

    return test_accuracy

def cifar_cnn_fitbo(X):
    # same as gpyopt_objective, except X is size (num_iter, input_dim)
    y = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        y[i] = cifar_cnn_gpyopt(X[i])

    return y
