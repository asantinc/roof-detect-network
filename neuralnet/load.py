import os
import shutil
import glob
import cv2
import pdb
import numpy as np

import matplotlib.pyplot as plt
import sklearn.utils
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

import experiment_settings as settings
from timer import do_cprofile

class RoofLoader(object):
    def roof_type(self, y):
        if y == 1:
            return 'Metal'
        elif y == 2:
            return 'Thatch'
        elif y == 0:
            return 'No roof'


    def load(self, max_roofs=None, roofs_only=False, test_percent=0.10, non_roofs=1):
        """OLD VERSION
        Load the data to a numpy array, return dataset divided into training and testing sets

        Parameters:
        roofs_only: If roofs_only is true we discard the non-roofs from the data
        non_roofs: the proportion of non_roofs relative to roof patches to include in the dataset
        """
        #get the labels   
        labels_list = np.loadtxt(open(settings.FTRAIN_LABEL,"rb"),delimiter=",")
        labels_dict = dict(labels_list)
        
        data_stats = np.bincount(labels_dict.values())
        total_roofs = data_stats[1]+data_stats[2]
        print 'DATA STATS:' 
        print data_stats
        
        #the non_roofs always come after, we take the roof labels and the proportion of non-roofs we want
        labels_list = labels_list[:(total_roofs+(non_roofs*total_roofs))]
        if max_roofs is None:
            max_roofs = len(labels_list)

        X, labels = self.load_images(labels_list[:max_roofs], max_roofs)
        labels = np.array(labels, dtype=np.int32)
        
        X, labels = sklearn.utils.shuffle(X, labels, random_state=42)  # shuffle train data    

        #remove the non_roofs
        if roofs_only:
            roof = (labels[:]>0)
            labels = labels[roof]
            X = X[roof]
            labels[labels[:]==2] = 0 
        
        self.X_train, self.X_test, self.labels_train, self.labels_test = cross_validation.train_test_split(X, labels, test_size=test_percent, random_state=0)
        return self.X_train, self.X_test, self.labels_train, self.labels_test


    def reduce_label_numbers(self):
        labels_list = np.loadtxt(open(settings.FTRAIN_LABEL,"rb"),delimiter=",")       
        #the non_roofs always come after, we take the roof labels and the proportion of non-roofs we want
        labels_list = labels_list[:130000]

        for f_name, roof_type in labels_list:
            f_number = int(f_name)
            f_path = settings.FTRAIN+str(f_number)+'.jpg'                
            shutil.copyfile(f_path, '../data/reduced_training/'+str(f_number)+'.jpg' )

    '''
    class DataScaler(StandardScaler):
    #Subclass of sklearn.StandardScaler that reshapes data as needed and then calls super to do scaling
    #
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
    '''



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

    
    def load_images(self, labels_tuples, max_roofs):
        '''
        Load the images into a numpy array X.
        '''
        X = np.empty((max_roofs,3,settings.PATCH_W, settings.PATCH_H))
        file_names = list()
        img_to_remove = list()
        labels = list()
        failures = 0

        index = 0
        for i, (f_name, roof_type) in enumerate(labels_tuples):
            if i%1000 == 0:
                print 'Loading image {0}'.format(i)
            f_number = int(f_name)
            f_path = settings.FTRAIN+str(f_number)+'.jpg'                
            x = cv2.imread(f_path)
            x = np.asarray(x, dtype='float32')/255
            try:
                x = x.transpose(2,0,1)
                x.shape = (1,x.shape[0], x.shape[1], x.shape[2])
                #X = x if X==None else np.concatenate((X, x), axis=0)
                X[index, :, :, :] = x
            except ValueError, e:
                print e
                failures += 1
                print 'fail:'+ str(failures)
            else:
                index += 1
                file_names.append(f_number)
                labels.append(roof_type)
        X = X.astype(np.float32)
        return X[:index, :,:,:], labels



###################################################
## Load image patches for neural network.
# 1. Loads positives and negatives (non_roofs*number of positive)
# 2. The scaling has already been done in get_data
# 3. Here we do the division over 255 though
###################################################

    def neural_load_training(self, train_path=None, neg_path=None, max_roofs=None, non_roofs=None):
        '''
        Load all the positive training examples for the neural network and labels to a numpy array
        '''
        train_path = settings.TRAINING_NEURAL_POS if train_path is None else train_path
        neg_path = settings.TRAINING_NEURAL_NEG if neg_path is None else neg_path
        train_labels =  np.loadtxt(open(train_path+'labels.csv',"rb"),delimiter=",")
        X_train, y_train = self.get_neural_data(train_labels, train_path)
        
        neg_labels = np.loadtxt(open(neg_path+'labels.csv', 'rb'),delimiter=',')
        pdb.set_trace()
        X_neg_train, y_neg_train = self.get_neural_data(neg_labels[:y_train.shape[0]*non_roofs],neg_path)
       
        print X_train.shape, y_train.shape
        print X_neg_train.shape, y_neg_train.shape
        pdb.set_trace()
        X = np.concatenate((X_train, X_neg_train), axis=0)
        labels = np.concatenate((y_train, y_neg_train), axis=0)

        #shuffle data
        X, labels = sklearn.utils.shuffle(X, labels, random_state=42)  # shuffle train data    
        return X, labels


    def get_neural_data(self, label_tuples, path):
        X = np.empty((len(label_tuples), 3, settings.PATCH_W, settings.PATCH_H))
        labels = list()
        failures = 0
        index = 0

        for i, (f_name, roof_type) in enumerate(label_tuples):
            if i%1000 == 0:
                print 'Loading image {0}'.format(i)
            f_number = int(f_name)
            f_path = path+str(f_number)+'.jpg'                
            
            x = cv2.imread(f_path)
            x = np.asarray(x, dtype='float32')/255
            try:
                x = x.transpose(2,0,1)
                x.shape = (1,x.shape[0], x.shape[1], x.shape[2])
                X[index, :, :, :] = x
            except ValueError, e:
                print e
                failures += 1
                print 'fail:'+ str(failures)
            else:
                index += 1
                labels.append(roof_type)
  
        #remove any failed images
        if failures > 0:
            X = X[:-failures, :,:,:] 
        

        #return the right type
        X = X.astype(np.float32)
        labels = np.array(labels).astype(np.int32)
        return X, labels


if __name__ == "__main__":
    loader = RoofLoader()
    loader.load(max_roofs=5000)

