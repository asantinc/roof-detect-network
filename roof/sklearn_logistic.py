import numpy as np
import load
import pdb

import sklearn.linear_model.logistic as logist
from sklearn import cross_validation, metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt 

import skimage
from skimage import color

OUT_PATH = '../output/'
IMG_SIZE = 40

def flatten_data(X):
    shape_new = 1
    itershape = iter(X.shape)
    next(itershape)
    for s in itershape:
        shape_new = shape_new*s
    return X.reshape(X.shape[0], shape_new)

def print_type(y):
    if y == 1:
        print 'Thatch'
    elif y == 2:
        print 'Metal'
    elif y == 1:
        print 'No roof'


def display_images(X, y=[], indeces=[0]):
    '''
    Display the images indicated by indices that are contained in X
    '''
    for i in indeces:
        print_type(y[i])
        if len(X.shape) == 2:
            if X.shape[1] > IMG_SIZE:
                x = np.reshape(X[i, :], (IMG_SIZE, IMG_SIZE))
        elif len(X.shape) == 3:
            x = X[i,:,:]
        else:
            x = np.squeeze(X[i,:,:,:])
            x = np.transpose(x, (1,2,0))
        plt.imshow(x)
        plt.show()


def sklearn_logistic(X, y, output=None):
    skf = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True)
    total_score = list()
    y_predictions = list()
    for train_index, test_index in skf:
        #scale the data
        scaler = preprocessing.StandardScaler().fit(X[train_index])
        X_train = scaler.transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        #apply logisitic regression
        log_reg = logist.LogisticRegression()
        score = (log_reg.fit(X_train,y[train_index]).score(X_test, y[test_index]))
        if output != None:
            output.write(str(score)+'\n')
            confuse = metrics.confusion_matrix(y[test_index], log_reg.predict(X[test_index])) 
            output.write(str(confuse)+'\n')

if __name__ == '__main__':
    #RGB: logistic to RGB image as a single column vector
    X, y, _ = load.load(roof_only=True)
    display_images(X, y, [0,1,2,3])
    X_flat = flatten_data(X)
    f =open(OUT_PATH+'rgb_logistic', 'w')
    sklearn_logistic(X_flat, y, f)
    f.close()

    #Grayscale: apply logistic regression to grayscale version of data
    X_gray = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for x in range(X.shape[0]):
        img = np.squeeze(X[x,::])
        X_gray[x,:,:] = color.rgb2gray(skimage.img_as_float(img)) 
    display_images(X_gray, [0,1,2,3])
    X_gray = flatten_data(X_gray) 
    f = open(OUT_PATH+'gray_logistic', 'w')   
    sklearn_logistic(X_gray, y, f)
    f.close()

    

