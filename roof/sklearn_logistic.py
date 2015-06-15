import numpy as np
import load
import pdb

import sklearn.linear_model.logistic as logist
from sklearn import cross_validation, metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt 

import skimage
from skimage import color

import experiment_settings as settings


def sklearn_logistic(X, y, out_file=None, k_fold=10):
    '''
    Use the built-in logistic regression from sklearn.
    Separate in k folds
    '''
    skf = cross_validation.StratifiedKFold(y, n_folds=k_fold, shuffle=True)
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
        if out_file != None:
            out_file.write(str(score)+'\n')
            confuse = metrics.confusion_matrix(y[test_index], log_reg.predict(X[test_index])) 
            out_file.write(str(confuse)+'\n')

if __name__ == '__main__':
    loader = load.RoofLoader()
    X, _, y, _, _ = loader.load(roof_only=True, test_percent=0)
    X_shape = X.shape
 
    #RGB: logistic to RGB image as a single column vector
    X_flat = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
    f = open(settings.OUT_PATH+'rgb_logistic', 'w')
    sklearn_logistic(X_flat, y, f)
    f.close()

    #Grayscale: apply logistic regression to grayscale version of data
    X_gray = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for x in range(X.shape[0]):
        img = np.squeeze(X[x,::])
        X_gray[x,:,:] = color.rgb2gray(skimage.img_as_float(img)) 
    X_shape = X_gray.shape
    X_gray = X_gray.reshape(X_shape[0], X_shape[1]*X_shape[2])
    
    f = open(settings.OUT_PATH+'gray_logistic', 'w')   
    sklearn_logistic(X_gray, y, f)
    f.close()

    

