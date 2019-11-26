import keras
import keras.backend as K
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


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


def build_model():
    """
    build a LeNet-style model
    2019/11/26      YANG Jie    Init
    :return: None
    """
    model = keras.Sequential()
    model.add(Conv1D(16, 5, strides=3, padding='same', input_shape=(300, 1)))
    model.add(MaxPool1D(2))
    model.add(Conv1D(32, 5, strides=3, padding='same'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(2))
    return model


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
    opt = keras.optimizers.sgd(lr=0.001)
    # set lr reduce scheduler
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    model.compile(loss=loss_func, optimizer=opt)
    # train the model
    hist = model.fit(X, Y, validation_split=0.2, epochs=100, verbose=2, batch_size=2048, callbacks=[reduce_lr])
    print('=' * 50)
    # predict the model
    Y_pre = model.predict(X_test, verbose=2)
    # show errors in graph
    errors = np.abs(Y_test - Y_pre)
    plt.plot(errors[:, 0])
    plt.plot(errors[:, 1])
    plt.legend(['P-err', 'R-err'])
    plt.show()
