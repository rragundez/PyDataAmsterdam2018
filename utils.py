import os
import random
import time

from datetime import timedelta
from itertools import groupby
from PIL import Image
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import layers
from keras.callbacks import Callback
from keras.models import Model
from keras.models import Sequential


def make_inceptionV3_custom_model(base_model):
    # get base model output
    x = base_model.output
    # add GlobalAveragePooling2D layer
    x = layers.GlobalAveragePooling2D(name='CustomLayer_1')(x)
    # add Dense layer of 512 units
    x = layers.Dense(name='CustomLayer_2', units=512, activation='relu')(x)
    # add output Dense layer with 10 units and softmax activation function
    predictions = layers.Dense(name='OutputLayer',
                               units=10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def save_keras_dataset_to_disk(X_train, y_train, X_test, y_test):
    for i, x_img in enumerate(X_train):
        label = y_train[i, 0]
        os.makedirs(f'data/CIFAR10/train/{label}', exist_ok=True)
        Image.fromarray(x_img).save(f"data/CIFAR10/train/{label}/{i}.jpeg")
    for i, x_img in enumerate(X_test):
        label = y_test[i, 0]
        os.makedirs(f'data/CIFAR10/test/{label}', exist_ok=True)
        Image.fromarray(x_img).save(f"data/CIFAR10/test/{label}/{i}.jpeg")


def make_CNN_model():
    model = Sequential()
    # input layer transformation (BatchNormalization + Dropout)
    model.add(layers.BatchNormalization(name='InputLayer',
                                        input_shape=(28, 28, 1)))
    model.add(layers.Dropout(name='Dropout_InputLayer', rate=0.3))

    # convolutional layer (Conv2D + MaxPooling2D + Flatten + Dropout)
    model.add(layers.Conv2D(name='ConvolutionalLayer_1',
                            filters=32,
                            kernel_size=(3, 3),
                            activation='relu',
                            border_mode="same"))
    model.add(layers.MaxPooling2D(name='MaxPooling_1'))
    model.add(layers.Flatten(name='Flatten_1'))
    model.add(layers.Dropout(rate=0.3))

    # fully connected layer (Dense + BatchNormalization + Activation + Dropout)
    model.add(layers.Dense(name='FullyConnectedLayer_2', units=150))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.3))

    # output layer (Dense + BatchNormalization + Activation)
    model.add(layers.Dense(name='OutputLayer', units=10))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))

    return model


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
