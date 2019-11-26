import keras
import keras.backend as K
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from ultis import *


def build_model():
    """
    build a LeNet-style model
    2019/11/26      YANG Jie    Init
    :return: None
    """
    model = keras.Sequential()
    model.add(Conv1D(16, 5, strides=3, padding='same', input_shape=(get_sample_length(), 1)))
    model.add(MaxPool1D(2))
    model.add(Conv1D(32, 5, strides=3, padding='same'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(2))
    return model


def lr_scheduler(epoch):
    """
    learning rate scheduler
    :param epoch:
    :return:
    """
    if epoch % 600 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.2)
        print('lr changed to {}'.format(lr * 0.2))
    return K.get_value(model.optimizer.lr)


if __name__ == '__main__':
    # load data
    X, Y = load_data('../data/segmentors.mat')
    # data normalization
    X = data_normalization(X)
    X, Y = data_shuffle(X, Y)
    # format X for keras input
    X = np.expand_dims(X, axis=2)
    # split set to train and test
    X_train, Y_train, X_test, Y_test = data_split(X, Y, radio=0.95)
    # build a model
    model = build_model()
    # print model information
    model.summary()
    # set learning rate and optimizer
    opt = keras.optimizers.adam(lr=1e-5)
    # set lr reduce scheduler
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    model.compile(loss=loss_func, optimizer=opt)
    # train the model
    hist = model.fit(X, Y, validation_split=0.1, epochs=1000, verbose=2, batch_size=2048, callbacks=[reduce_lr])
    print('=' * 50)
    # plot loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
    # predict the model
    Y_pre = model.predict(X_test, verbose=2)
    # show errors in graph
    errors = np.abs(Y_test - Y_pre)
    plt.plot(errors[:, 0])
    plt.plot(errors[:, 1])
    plt.legend(['P-err', 'R-err'])
    plt.show()
