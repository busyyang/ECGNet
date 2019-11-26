import scipy.io as scio
import keras.backend as K
import numpy as np


def load_data(path_file):
    """
    load data from .mat file. Preprocessed in matlab.
    2019/11/26      YANG Jie    Init
    :param path_file: segmentor.mat
    :return: data(X), ann(Y)
    """
    file = scio.loadmat(path_file)
    data = file['segs']
    ann = file['anns']
    return data, ann


def loss_func(y_true, y_pre):
    """
    user loss function in equation of RMSE
    2019/11/26      YANG Jie    Init
    :param y_true:
    :param y_pre:
    :return:
    """
    return K.sqrt(K.mean(K.pow(y_true - y_pre, 2)))


def data_normalization(data):
    """
    normalization data in row direction.
    return a output with mean of 0 and std of 1
    2019/11/26      YANG Jie    Init
    :param data:
    :return:
    """
    data_mean = data.mean(axis=1)
    data_mean = data_mean.reshape(data_mean.shape[0], 1)
    data_std = data.std(axis=1)
    data_std = data_std.reshape(data_std.shape[0], 1)
    return (data - data_mean) / data_std


def data_shuffle(X, Y):
    """
    shuffle data. Get random order output.
    2019/11/26      YANG Jie    Init
    :param X:
    :param Y:
    :return:
    """
    per = np.random.permutation(len(Y))
    X = X[per, :]
    Y = Y[per, :]
    return X, Y


def data_split(X, Y, radio=0.7):
    """
    split set into train set and test set according to radio
    :param X:
    :param Y:
    :param radio:
    :return:
    """
    length = Y.shape[0]
    X_train = X[:int(length * radio), :]
    Y_train = Y[:int(length * radio), :]
    X_test = X[int(length * radio):, :]
    Y_test = Y[int(length * radio):, :]
    return X_train, Y_train, X_test, Y_test



