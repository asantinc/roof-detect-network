from collections import OrderedDict
from datetime import datetime
from functools import reduce
import numpy as np
import operator
import sys
import pdb

import lasagne
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
from nolearn.lasagne.handlers import PrintLog, PrintLayerInfo 
from nolearn.lasagne.util import is_conv2d

#my modules
import load
from sklearn.metrics import classification_report, confusion_matrix

#Constants for patch production
PATCHES_OUT_PATH = '../data/debug/'
LABELS_PATH = '../data/debug/labels.csv'
INHABITED_PATH = '../data/inhabited/'
UNINHABITED_PATH = '../data/uninhabited/'
DELETE_PATH = '../data/delete/'
NEGATIVE_PATCHES_NUM = 20

#types of roof
NON_ROOF = 0
METAL = 1
THATCH = 2

#Constants for image size
IMG_SIZE = 40
CROP_SIZE = 32
PATCH_W = PATCH_H = 40

CASCADE_PATH = '../viola_jones/cascades/'

#Constants for training neural network
OUT_REPORT = "../output/report/" 
OUT_HISTORY = "../output/history/"
OUT_IMAGES = "../output/images/"
FTRAIN = '../data/training/'
FTRAIN_LABEL = '../data/training/labels.csv'
TEST_PATH = '../data/test/'

#Constants for debugging
VERBOSITY = 1   #varies from 1 to 3
DEBUG = False

#Viola constants
BG_FILE = '../viola_jones/bg.txt'
DAT_PATH = '../viola_jones/all_dat/'
VEC_PATH = '../viola_jones/vec_files/'
VIOLA_AUGM_DATA = '../viola_jones/data/'
CASCADE_PATH = '../viola_jones/cascades/'

def print_debug(to_print, verbosity=1):
    #Print depending on verbosity level
    
    if verbosity <= VERBOSITY:
        print str(to_print)


class Experiment(object):
    def __init__(self, net=None, test_percent=.10, printer=None, non_roofs=2, roofs_only=False):
        self.net=net
        self.test_percent=test_percent
        self.printer=printer
        self.non_roofs=non_roofs    #the proportion of non_roofs relative to roofs to be used in data
        self.roofs_only=roofs_only

    def train_test(self):
        '''
        Train and test neural network. Also print out evaluation.
        '''
        #save settings to file
        self.printer.log_to_file(self.net, self.__str__(), overwrite=True)
        
        #load data
        self.roof_loader = load.RoofLoader()
        X_train, X_test, y_train, y_test = self.roof_loader.load(test_percent=self.test_percent, non_roofs=self.non_roofs, roofs_only=self.roofs_only)
        
        #rescale X_train and X_test
        self.scaler = load.DataScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform2(X_test)
    
        #fit the network to X_train
        self.net.fit(X_train, y_train)
        self.net.save_weights()

        #find predictions for test set
        predicted = self.net.predict(X_test)
        self.evaluation(predicted, X_train, X_test, y_train, y_test)


    def test_preloaded_single(self, test_case):
        if self.scaler is None:
            self.net.load_params_from('saved_weights/'+self.net.net_name+'.pickle')
            
            #we need to get the training set so we can get the right scaling for the test set
            self.roof_loader = load.RoofLoader()
            X_train, X_test, y_train, y_test = self.roof_loader.load(test_percent=self.test_percent, 
                    non_roofs=self.non_roofs, roofs_only=self.roofs_only)
            
        #rescale X_train and X_test, using information from the training set only
        self.scaler = load.DataScaler()
        self.scaler.fit_transform(X_train)
        test_case = self.scaler.trasform(test_case)
        return self.net.predict(X_test)


    def test_preloaded(self, plot_loss=True, test_case=None):  
        '''Preload weights, classify roofs and write evaluation
        To classify a single instance it must be passed in as the test_case parameter. Otherwise,
        the method will test the network on the test portion of the training set
        '''
        #save settings to file
        self.printer.log_to_file(self.net, self.__str__(), overwrite=True)
        self.roof_loader = load.RoofLoader()
        
        #we need to get the training set so we can get the right scaling for the test set
        X_train, X_test, y_train, y_test = self.roof_loader.load(test_percent=self.test_percent, 
                non_roofs=self.non_roofs, roofs_only=self.roofs_only)
        X_test = test_case if test_case is not None else X_test

        #rescale X_train and X_test, using information from the training set only
        if self.scaler:
            scaler = load.DataScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform2(X_test)

        #find predictions for test set
        self.net.load_params_from('saved_weights/'+self.net.net_name+'.pickle')
        predicted = self.net.predict(X_test)
        self.evaluation(predicted, X_train, X_test, y_train, y_test)


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
            mistaken_imgs = scaler.inverse_transform(mistaken_imgs)
            roof_loader.display_images(mistaken_imgs, labels=y_test[mistakes], indeces=range(len(mistaken_imgs)))


    def __str__(self):
        out_list = list()
        for key, value in self.__dict__.items():
            out_list.append(str(key)+': '+str(value))
        out_list.append('\n')
        return '\n'.join(out_list)          



class PrintLogSave(PrintLog):
    def __call__(self, nn, train_history): 
        file = open(OUT_HISTORY+nn.net_name, 'a')
        file.write(self.table(nn, train_history))
        file.close()

    def table(self, nn, train_history):
        info = train_history[-1]
        return str(info['epoch'])+'\t'+ str(info['train_loss'])+ '\t'+str(info['valid_loss'])+'\t'+str( info['train_loss'] / info['valid_loss'])+'\t'+str(info['valid_accuracy'])+'\n'
            
    def log_to_file(self, nn, log, overwrite=False, binary=False, title=''):
        write_type = 'w' if overwrite else 'a'
        file = open(OUT_REPORT+nn.net_name, write_type)
        file.write(title)
        if binary:
            print >> file, log
        else:
            file.write(log)
        file.close()


class SaveLayerInfo(PrintLayerInfo):
    def __call__(self, nn, train_history):
        file = open(OUT_REPORT+nn.net_name, 'a')
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
            non_roofs=int(float(arg))
        elif opt=='-p':
            preloaded=bool(arg)
        elif opt=='-l':
            num_layers=int(float(arg))
        elif opt=='-r':
            roofs_only=True
        elif opt=='-a':
            net_name=arg
        elif opt=='-e':
            epochs=int(float(arg))
    return test_percent, non_roofs, preloaded, num_layers, roofs_only, plot, net_name, epochs


if __name__ == '__main__':
    test_percent, non_roofs, preloaded, num_layers, roofs_only, plot, net_name, epochs = set_parameters()  
    log = True
    plot_loss = True
    name_percent = str(int(100*test_percent))
    net_name = 'conv'+str(num_layers)+'_nonroofs'+str(non_roofs)+'_test'+name_percent if net_name==None else net_name
    if roofs_only:
        net_name = net_name+'_roofs'
     
    layers = MyNeuralNet.produce_layers(num_layers)      
    printer = PrintLogSave()
    #set up the experiment
    experiment = Experiment(data_augmentation=True,
                    test_percent=test_percent,
                    scaler='StandardScaler',
                    preloaded=preloaded,
                    printer=printer,
                    display_mistakes=True,
                    non_roofs=non_roofs,
                    roofs_only=roofs_only
                    )
    experiment.net = MyNeuralNet(
        layers=layers,
        input_shape=(None, 3, settings.CROP_SIZE, settings.CROP_SIZE),
        output_num_units=3,
        
        output_nonlinearity=lasagne.nonlinearities.softmax,
        preproc_scaler = None, 
        
        #learning rates
        update_learning_rate=0.01,
        update_momentum=0.9,
        
        #printing
        net_name=net_name,
        on_epoch_finished=[printer],
        on_training_started=[SaveLayerInfo()],

        #data augmentation
        batch_iterator_test=flip.CropOnlyBatchIterator(batch_size=128),
        batch_iterator_train=flip.FlipBatchIterator(batch_size=128),
        
        max_epochs=epochs,
        verbose=1,
        ) 
    experiment.net.set_layer_params(num_layers)
    experiment.run(log=log, plot_loss=plot_loss) 

