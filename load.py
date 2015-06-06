import os
import glob
import numpy as np
import cv2
import pdb
#from pandas.io.parsers import read_csv
#from sklearn.utils import shuffle


FTRAIN = 'data/train/'
FTRAIN_LABEL = 'data/labels.csv'
FTEST = 'data/test/'


def load_images(test=False):
    fname = FTEST if test else FTRAIN
    X = None
    ignore_rows = list()
    for f in glob.glob(fname+'*.jpg'):
        file_number = f[11:-4]
        x = cv2.imread(f)
        total_shape = x.shape[0]*x.shape[1]*x.shape[2]
        x.shape = (1,total_shape)
        try:
            X = x if X==None else np.concatenate((X, x), axis=0)
            #print X.shape, x.shape
        except ValueError, e:
            #pdb.set_trace()
            ignore_rows.append(int(file_number))
            #print X.shape, x.shape
            print e
    X = X.astype(np.float32)
    return X, ignore_rows


def load(test=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    """
    X, ignore_rows = load_images(test)

    #get the labels
    if not test:  # only FTRAIN has any target columns
        y = np.loadtxt(open(FTRAIN_LABEL,"rb"),delimiter=",", usecols=[1])
        #pdb.set_trace()
        y = np.delete(y, ignore_rows, axis=0)
        #X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.int32)
    else:
        y = None
    print X.shape, y.shape
    #pdb.set_trace()
    return X, y

if __name__ == "__main__":
    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))
