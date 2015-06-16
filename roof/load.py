import os
import glob
import cv2
import pdb
import matplotlib.pyplot as plt
import numpy as np
import sklearn.utils
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

import experiment_settings as settings

class RoofLoader(object):
    def roof_type(self, y):
        if y == 1:
            return 'Metal'
        elif y == 2:
            return 'Thatch'
        elif y == 0:
            return 'No roof'

    def display_images(self, X, labels=[], indeces=[0], file_names=[]):
        '''
        Display the images indicated by indeces that are contained in X
        '''
        for i in indeces:
            if len(labels)==len(X):
                print self.roof_type(labels[i])
            if len(X.shape) == 2:
                if X.shape[1] > settings.IMG_SIZE:
                    x = np.reshape(X[i, :], (settings.IMG_SIZE, settings.IMG_SIZE))
            elif len(X.shape) == 3:
                x = X[i,:,:]
            else:
                x = np.squeeze(X[i,:,:,:])
                x = np.transpose(x, (1,2,0))*255
            plt.imshow(x)
            plt.show()

    def display_single(self, other_info=""):
        print other_info+'\n'
        plt.imshow(img)
        plt.show()

    def load_images(self):
        '''
        Load the images into a numpy array X.
        '''
        X = None
        file_names = list()
        for f in glob.glob(settings.FTRAIN+'*.jpg'):
            x = cv2.imread(f)
            x = np.asarray(x, dtype='float32')/255
            x = x.transpose(2,0,1)
            x.shape = (1,x.shape[0], x.shape[1], x.shape[2])

            try:
                X = x if X==None else np.concatenate((X, x), axis=0)
                file_number = f[len(settings.FTRAIN):-4]
                file_names.append(int(file_number))
            #some images are not loading properly because they are smaller than expected
            except ValueError, e:
                print e
        
        X = X.astype(np.float32)
        return X, file_names


    def load(self, roof_only=False, test_percent=0.10):
        """
        If roof_only is true we discard the non-roofs from the data
        """
        X, self.file_names = self.load_images()
        #get the labels   
        labels_list = np.loadtxt(open(settings.FTRAIN_LABEL,"rb"),delimiter=",")
        labels_dict = dict(labels_list)
        labels = []
        
        #only get the labels for the images that we have loaded properly
        for i, f in enumerate(self.file_names):
            f_int = int(f)
            if f_int in labels_dict:
                labels.append(int(labels_dict[f_int]))
            
        labels = np.array(labels, dtype=np.int32)
        X, labels, self.file_names = sklearn.utils.shuffle(X, labels, self.file_names, random_state=42)  # shuffle train data    
        
        if roof_only:
            roof = (labels[:]>0)
            labels = labels[roof]
            X = X[roof]
            labels[labels[:]==2] = 0 
        #data_stats = np.bincount(labels)
        self.X_train, self.X_test, self.labels_train, self.labels_test = cross_validation.train_test_split(X, labels, test_size=test_percent, random_state=0)
        return self.X_train, self.X_test, self.labels_train, self.labels_test, self.file_names

class DataScaler(StandardScaler):
    '''Subclass of sklearn.StandardScaler that reshapes data as needed and then calls super to do scaling
    '''
    def fit_transform(self, X):
        X_shape = X.shape
        X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
        X = super(DataScaler, self).fit_transform(X)
        return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])

    def transform2(self, X):
        X_shape = X.shape
        X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
        X = super(DataScaler, self).transform(X)
        return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])

    def inverse_transform(self, X): 
        X_shape = X.shape
        X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
        X = super(DataScaler, self).inverse_transform(X)
        return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, file_names = load()
    display_images(X, y, indeces=[1,2,3,4,5,6,7,8,9,10], file_names=file_names)

