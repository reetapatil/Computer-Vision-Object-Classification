import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
import glob
import cv2 as cv
import os

from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, SGD

from keras import backend as K, regularizers, metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

CACHE_DIR = "cache"


def train_svm(x_train, y_train, kernel):
    model_file = CACHE_DIR + os.sep + "svm_" + kernel + ".joblib"

    if os.path.exists(model_file):
        print("Loading model...")
        model = joblib.load(model_file)
        print("Loaded!")
    else:
        print("Preprocessing data...")
        x_train = [cv.cvtColor(img.astype('float32'),
                               cv.COLOR_BGR2GRAY).flatten()
                   for img in x_train]
        model = SVC(kernel=kernel, C=1.0)
        print("Training model...")
        model.fit(x_train, y_train)
        print("Done!")
        joblib.dump(model, model_file)
    return model


def evaluate_model(trained_model, x_test, y_test):
    print("Preprocessing data...")

    x_test = [cv.cvtColor(img.astype('float32'),
                          cv.COLOR_BGR2GRAY).flatten()
              for img in x_test]
    print("Predicting!")
    return trained_model.score(x_test, y_test)
