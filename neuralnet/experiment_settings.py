from collections import OrderedDict
import os
import subprocess
from datetime import datetime
from functools import reduce
import numpy as np
import operator
import sys
import pdb
import csv
from operator import itemgetter
import cv2

import lasagne
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
from nolearn.lasagne.handlers import PrintLog, PrintLayerInfo 
from nolearn.lasagne.util import is_conv2d
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform as sp_rand
from scipy.stats import randint
from sklearn.grid_search import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import skimage
import datetime

#my modules
import load
import my_net
import FlipBatchIterator as flip
import utils

class Experiment(object):
    def __init__(self, test_path=None, train_path=None, flip=True, preloaded_path=None, pipeline=False, 
                    print_out=True, preloaded=False, test_percent=.20, non_roofs=2, 
                    roofs_only=False,log=True, plot_loss=True, plot=True, epochs=50, net_name=None, num_layers=None, max_roofs=None):
        self.pipeline = pipeline
        self.num_layers = num_layers
        self.net_name = net_name

        self.epochs = epochs
        self.test_percent = test_percent
        self.non_roofs = non_roofs    #the proportion of non_roofs relative to roofs to be used in data
        self.roofs_only = roofs_only
        self.log = log
        self.plot_loss = True
        self.flip = flip
        self.test_path = test_path
        self.train_path = train_path

        print 'Setting up the Neural Net \n'
        self.setup_net(print_out=print_out)

        #preload weights if a path to weights was provided
        self.preloaded_path = None
        if preloaded_path is not None:
            preloaded_path = preloaded_path if preloaded_path.endswith('.pickle') else preloaded_path+'.pickle'
            self.preloaded_path = utils.get_path(neural=True, params=True)+preloaded_path
            self.net.load_params_from(self.preloaded_path)      
            #TODO: need to load the scaler of the data as well as the parameters!!!


        if self.pipeline==False: #if pipeline is true, we don't need to load data, testing patches will be passed in 
            print 'Loading data \n'
            self.roof_loader = load.RoofLoader()

            self.X, self.y = self.roof_loader.neural_load_training(non_roofs=non_roofs)
            print self.X.shape, self.y.shape
            print 'Data is loaded \n'

            #set up the data scaler
            self.scaler = DataScaler()
            self.X = self.scaler.fit_transform(self.X)
            #self.X_test =  self.scaler.transform2(self.X_test)



    def setup_net(self, print_out=True):
        if print_out:
            self.printer = PrintLogSave()
            on_epoch_finished = [self.printer]
            on_training_started = [SaveLayerInfo()]
        else:
            on_epoch_finished = []
            on_training_started = []

        layers, layer_params = my_net.MyNeuralNet.produce_layers(self.num_layers)      
        self.net = my_net.MyNeuralNet(
            layers=layers,
            num_layers=self.num_layers,
            input_shape=(None, 3, utils.CROP_SIZE, utils.CROP_SIZE),
            output_num_units=3,

            output_nonlinearity=lasagne.nonlinearities.softmax,
            preproc_scaler = None, 
        
            #learning rates
            update_learning_rate=0.01,
            update_momentum=0.9,
        
            #printing
            net_name=self.net_name,
            on_epoch_finished=on_epoch_finished,
            on_training_started=on_training_started,

            #data augmentation
            #batch_iterator_test= flip.CropOnlyBatchIterator(batch_size=128),
            batch_iterator_train=flip.FlipBatchIterator(batch_size=128),

        
            max_epochs=self.epochs,
            verbose=1,
            **layer_params
            )


    def train_test(self):
        '''
        Train and test neural network. Also print out evaluation.
        '''
        #save settings to file
        self.printer.log_to_file(self.net, self.__str__(), overwrite=True)
    
        #fit the network to X_train
        self.net.fit(self.X, self.y)
        self.net.save_weights()

        #find predictions for test set
    
        raise ValueError('Does this prediction do it with 40x40 crops of 32x32 crops?') 
        #predicted = self.net.predict(self.X_test)
        #self.evaluation(predicted, self.X_train, self.X_test, self.y_train, self.y_test)


    def test(self, test_patches): 
        return self.net.predict(test_patches)

    def test_preloaded(self, plot_loss=True, test_case=None):  
        '''Preload weights, classify roofs and write evaluation
        To classify a single instance it must be passed in as the test_case parameter. Otherwise,
        the method will test the network on the test portion of the training set
        '''
        #save settings to file
        self.printer.log_to_file(self.net, self.__str__(), overwrite=True)
    
        #find predictions for test set
        #need to reduce it from 40 pixels down to 32 pixels
        min = (utils.PATCH_H - utils.CROP_SIZE)/2
        print self.X_test.shape
        self.X_test = self.X_test[:, :, min:min+utils.CROP_SIZE, min:min+utils.CROP_SIZE]
        print self.X_test.shape
        predicted = self.net.predict(self.X_test)
        self.evaluation(predicted, self.X_train, self.X_test, self.y_train, self.y_test)


    def evaluation(self, predicted, X_train, X_test, y_train, y_test):
        #print evaluation
        self.printer.log_to_file(self.net, confusion_matrix(y_test, predicted), binary=True, title='\n\nConfusion Matrix\n')
        self.printer.log_to_file(self.net, classification_report(y_test, predicted), title='\n\nReport\n')
    
        #save a plot of the validation and training losses
        #if self.plot_loss:
        self.net.save_loss()
    
        #display mistakes
        if self.display_mistakes: 
            mistakes = np.array([True if y_test[i]-predicted[i] != 0 else False for i in range(len(y_test))])
            mistaken_imgs = X_test[mistakes]
            mistaken_imgs = self.scaler.inverse_transform(mistaken_imgs)
            roof_loader.display_images(mistaken_imgs, labels=y_test[mistakes], indeces=range(len(mistaken_imgs)))


    @staticmethod
    def report_grid(grid_scores, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i , score in enumerate(top_scores):
            print "Model with rank: {0}".format(i+1)
            print "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    score.mean_validation_score,
                    np.std(score.cv_validation_scores))
            print "Parameters: {0}".format(score.parameters)
            print ""

    def optimize_params(self):
        params_grid = {"update_learning_rate": sp_rand(0.001,0.01), "momentum":sp_rand(0.9,2.0), "epochs":randint(50,300)}
        rsearch = RandomizedSearchCV(estimator=self.net, param_distributions=params_grid, n_iter=15, n_jobs=1)
        X, _, y, _ = self.roof_loader.load(max_roofs=100, test_percent=0, non_roofs=self.non_roofs, roofs_only=self.roofs_only)
        rsearch.fit(X,y)
        Experiment.report_grid(rsearch.grid_scores_)


    def __str__(self):
        out_list = list()
        for key, value in self.__dict__.items():
            out_list.append(str(key)+': '+str(value))
        out_list.append('\n')
        return '\n'.join(out_list)          


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
        single_image = False

        #reshape for transformation
        if len(X.shape) == 3:
            single_image = True
            X = X[None, :,:,:] 
            X = X.reshape(1, X_shape[0]*X_shape[1]*X_shape[2])
        else:
            X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
    
        X = super(DataScaler, self).transform(X)
    
        if single_image:
            X = np.squeeze(X)
            return X.reshape(X_shape[0], X_shape[1], X_shape[2])
        else:
            return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])

    def inverse_transform(self, X): 
        X_shape = X.shape
        X = X.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
        X = super(DataScaler, self).inverse_transform(X)
        return X.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])


class PrintLogSave(PrintLog):
    def __call__(self, nn, train_history): 
        file = open(utils.get_path(in_or_out=utils.OUT, neural=True, data_fold=utils.TRAINING)+nn.net_name+'_history', 'a')
        file.write(self.table(nn, train_history))
        file.close()

    def table(self, nn, train_history):
        info = train_history[-1]
        return str(info['epoch'])+'\t'+ str(info['train_loss'])+ '\t'+str(info['valid_loss'])+'\t'+str( info['train_loss'] / info['valid_loss'])+'\t'+str(info['valid_accuracy'])+'\n'
        
    def log_to_file(self, nn, log, overwrite=False, binary=False, title=''):
        write_type = 'w' if overwrite else 'a'
        file = open(utils.get_path(in_or_out=utils.OUT, neural=True, data_fold=utils.TRAINING)+nn.net_name, write_type)
        file.write(title)
        if binary:
            print >> file, log
        else:
            file.write(log)
        file.close()


class SaveLayerInfo(PrintLayerInfo):
    def __call__(self, nn, train_history):
        file = open(utils.get_path(in_or_out=utils.OUT, neural=True, data_fold=utils.TRAINING)+nn.net_name, 'a')
        message = self._get_greeting(nn)
        file.write(message)
        file.write("## Layer information")
        file.write("\n\n")

        layers_contain_conv2d = is_conv2d(list(nn.layers_.values()))
        if not layers_contain_conv2d or (nn.verbose < 2):
            layer_info = self._get_layer_info_plain(nn)
            legend = None
        else:
            layer_info, legend = self._get_layer_info_conv(nn)
        file.write(layer_info)
        if legend is not None:
            file.write(legend)
        file.write(" \n\n")

        file.close()


def set_parameters():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:n:l:p:r:a:e:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2) 
    test_percent=0.2
    non_roofs=1
    preloaded=False
    num_layers=0 #logistic
    roofs_only=True
    plot=True
    net_name=None
    epochs=250
    for opt, arg in opts:
        if opt == '-t':
            test_percent=float(arg)
        elif opt == '-n':
            num_layers = int(float(arg))
        elif opt=='-a':
            net_name=arg
        elif opt=='-e':
            epochs=int(float(arg))
    return test_percent, non_roofs, preloaded, num_layers, roofs_only, plot, net_name, epochs


def get_params_from_file(file_name):
    parameters = dict()
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for par in reader:
            if len(par) == 2: 
                if par[0].strip() == 'test_percent':
                    parameters[par[0].strip()] = (float(par[1].strip()))
                elif par[0].strip() == 'test_path' or par[0].strip() == 'train_path':
                    parameters[par[0].strip()] = par[1].strip()
                else:
                    parameters[par[0].strip()] = int(float(par[1].strip()))
    return parameters

if __name__ == '__main__':
    param_file = 'params'+raw_input('Enter param file number :')+'.csv'
    params = get_params_from_file(NET_PARAMS_PATH+param_file) 

    if params['net_name'] == 0:
        params['net_name'] = time_stamped(param_file)
        print 'Network name is: {0}'.format(params['net_name'])
    if params['roofs_only']:
        params['net_name'] = params['net_name']+'_roofs'


    to_do = 't'
    #to_do = raw_input('o to optimize, t to train: ')
    #set up the experiment
    experiment = Experiment(print_out=True, **params)
    if to_do == 'o':
        experiment.optimize_params()
    elif to_do == 't':
        experiment.train_test() 
    else:
        img = cv2.imread("../data/inhabited/0001.jpg")
        experiment.test_image(img)


