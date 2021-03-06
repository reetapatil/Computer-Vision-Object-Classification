import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
import glob
import cv2
import os

from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, SGD

from keras import backend as K, regularizers, metrics


CACHE_DIR = "cache" + os.sep


def get_model(model_num, size, num_outputs):
    model_dictionary = {
        0: [
            Conv2D(16, (3, 3), activation='relu', input_shape=size),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2()),
            Dropout(0.5),
            Dense(num_outputs, activation='softmax')
        ],
        1: [
            Conv2D(32, (3, 3), activation='relu', input_shape=size),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2()),
            Dropout(0.5),
            Dense(num_outputs, activation='softmax')
        ],
        2: [
            Conv2D(64, (3, 3), activation='relu', input_shape=size),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2()),
            Dropout(0.5),
            Dense(num_outputs, activation='softmax')
        ],
    }
    model = Sequential()
    for layer in model_dictionary[model_num]:
        model.add(layer)

    model.compile(optimizer="Adam", loss="categorical_crossentropy",
                  metrics=[metrics.categorical_accuracy])

    return model


def train_convolutional(model_num, size, num_outputs, x_train, y_train,
                        epochs=10, batch_size=128, verbose=0):
    try:
        print("Loading model...")
        trained_model = keras.models.load_model(CACHE_DIR + os.sep +
                                                "trained_convolutional_" +
                                                str(model_num))
    except OSError:
        print("Training!")
        # One Hot Encode the Output
        y_train = keras.utils.to_categorical(y_train)

        trained_model = get_model(model_num, size, num_outputs)
        trained_model.fit(x_train, y_train, batch_size=batch_size,
                          epochs=epochs, verbose=verbose)

        keras.models.save_model(trained_model, CACHE_DIR +
                                os.sep + "trained_convolutional_" + str(
            model_num))

    return trained_model


def evaluate_model(trained_model, x_test, y_test):
    y_test = keras.utils.to_categorical(y_test)
    score = trained_model.evaluate(x_test, y_test, verbose=0)
    return score
