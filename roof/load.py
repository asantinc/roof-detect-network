import os
import glob
import cv2
import pdb
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils

FTRAIN = '../data/train/'
FTRAIN_LABEL = '../data/labels.csv'
FTEST = '../data/test/'
IMG_SIZE = 40

def load_images(test=False):
    '''
    Load the images into a numpy array X.
    We divide pixels by 255 and then subtract the mean from each image.
    '''
    fname = FTEST if test else FTRAIN
    X = None
    ignore_rows = list()
    
    for f in glob.glob(fname+'*.jpg'):
        file_number = f[len(FTRAIN):-4]
        x = cv2.imread(f)
        x = np.asarray(x, dtype='float32')/255
        x = x.transpose(2,0,1)
        x.shape = (1,x.shape[0], x.shape[1], x.shape[2])

        try:
            X = x if X==None else np.concatenate((X, x), axis=0)
        #some images are not loading properly because they are smaller than expected
        except ValueError, e:
            ignore_rows.append(int(file_number))
            print e
    X = X.astype(np.float32)
    return X, ignore_rows


def load(test=False, roof_only=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
       If roof_only is true we discard the non-roofs from the data
    """
    X, ignore_rows = load_images(test)
    data_stats = []
    #get the labels  
    
    if not test:  # only FTRAIN has any target columns
        y = np.loadtxt(open(FTRAIN_LABEL,"rb"),delimiter=",", usecols=[1])
        y = np.delete(y, ignore_rows, axis=0)
        X, y = sklearn.utils.shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.int32)
        if roof_only:
            roof = (y[:]>0)
            y = y[roof]
            X = X[roof]
            y[y[:]==2] = 0 
        data_stats = np.bincount(y)
    else:
        y = None
    
    return X, y, data_stats


if __name__ == "__main__":
    X, y, data_stats = load(test=True)
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))
    plt.imshow(X[1,:,:,:])
    plt.show()

