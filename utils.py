import os
import random
import time

from datetime import timedelta
from itertools import groupby
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import layers
from keras.callbacks import Callback
from keras.models import Sequential


def make_overkill_model():
    model = Sequential()
    # input layer transformation
    model.add(layers.BatchNormalization(name='InputLayer', input_shape=(2,)))
    model.add(layers.Dropout(0.3))

    # 1st hidden
    model.add(layers.Dense(name='HiddenLayer_1', units=30))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.Dropout(0.3))

    # 2nd hidden
    model.add(layers.Dense(name='HiddenLayer_2', units=10))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))

    # 3rd hidden
    model.add(layers.Dense(name='HiddenLayer_3', units=3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))

    # output layer
    model.add(layers.Dense(name='OutputLayer', units=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))

    return model


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '123'
    random.seed(123)
    np.random.seed(123)
    tf.set_random_seed(123)


def plot_training_summary(training_summary, time_summary=None):
    if time_summary:
        print('Training time: '
              f'{timedelta(seconds=time_summary.training_time)}(HH:MM:SS)')
        print('Epoch time avg: '
              f'{timedelta(seconds=mean(time_summary.epoch_times))}(HH:MM:SS)')
    hist = sorted(training_summary.history.items(),
                  key=lambda x: (x[0].replace('val_', ''), x[0]))

    epochs = [e + 1 for e in training_summary.epoch]
    for metric, values in groupby(hist,
                                  key=lambda x: x[0].replace('val_', '')):
        if 'val_loss' in training_summary.history:
            val0, val1 = tuple(values)
            plt.plot(epochs, val0[1], epochs, val1[1], '--', marker='o')
        else:
            val0 = tuple(values)[0]
            plt.plot(epochs, val0[1], '--', marker='o')
        plt.xlabel('epoch'), plt.ylabel(val0[0])
        plt.legend(('Train set', 'Validation set'))
        plt.show()


class TimeSummary(Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []
        self.training_time = time.process_time()

    def on_train_end(self, logs={}):
        self.training_time = time.process_time() - self.training_time

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.process_time()

    def on_epoch_end(self, batch, logs={}):
        self.epoch_times.append(time.process_time() - self.epoch_time_start)
