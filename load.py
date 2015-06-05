import os

import numpy as np
import cv2
#from pandas.io.parsers import read_csv
#from sklearn.utils import shuffle


FTRAIN = '~/roof/data/train/'
FTRAIN_LABEL = '~/roof/data/label.csv'
FTEST = '~/roof/data/test/'


def load_images(test=false):
    fname = FTEST if test else FTRAIN
    
    return X


def load(test=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    """
    fname = FTEST if test else FTRAIN
    X = load_images(test)

    #get the labels
    if not test:  # only FTRAIN has any target columns
        y = np.loadtxt(open("label.csv","rb"),delimiter=",",usecols=1)

        #X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))