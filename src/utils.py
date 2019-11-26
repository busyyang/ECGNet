import scipy.io as scio


def loadmatfile(filename):
    mat = scio.loadmat(filename)
    return mat


def loadann(filename):
    return scio.loadmat(filename)


if __name__ == '__main__':
    mat = loadmatfile('../data/sel30data.mat')
    ann = loadann('../data/sel30ann.mat')
