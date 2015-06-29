from collections import OrderedDict
from datetime import datetime
from functools import reduce
import numpy as np
import operator
import sys
import pdb
'''
import lasagne
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
from nolearn.lasagne.handlers import PrintLog, PrintLayerInfo 
from nolearn.lasagne.util import is_conv2d

#my modules
import load
from sklearn.metrics import classification_report, confusion_matrix
'''
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

def print_debug(to_print, verbosity=1):
    #Print depending on verbosity level
    
    if verbosity <= VERBOSITY:
        print str(to_print)

'''
class Experiment(object):
    def __init__(self, net=None, data_augmentation=True, display_mistakes=False, 
                test_percent=.10, scaler=True, preloaded=True, printer=None, non_roofs=2, roofs_only=False):
        self.net=net
        self.data_augmentation=data_augmentation
        self.test_percent=test_percent
        self.scaler=scaler
        self.preloaded=preloaded
        self.printer=printer
        self.display_mistakes=display_mistakes
        self.non_roofs=non_roofs    #the proportion of non_roofs relative to roofs to be used in data
        self.roofs_only=roofs_only

    def run(self, log=True, plot_loss=False):
        self.plot_loss=plot_loss
        #save settings to file
        if log:
            self.printer.log_to_file(self.net, self.__str__(), overwrite=True)

        #only train the network if we choose not to preload weights
        if self.preloaded:
            self.net.load_params_from('saved_weights/'+self.net.net_name+'.pickle')
        else:
            #load data
            roof_loader = load.RoofLoader()
            X_train, X_test, y_train, y_test = roof_loader.load(test_percent=self.test_percent, non_roofs=self.non_roofs, roofs_only=self.roofs_only)
            #rescale X_train and X_test
            if self.scaler:
                scaler = load.DataScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform2(X_test)
            #fit the network to X_train
            self.net.fit(X_train, y_train)
            self.net.save_weights()
        
        #find predictions for test set
        predicted = self.net.predict(X_test)

        #print evaluation
        self.printer.log_to_file(self.net, confusion_matrix(y_test, predicted), binary=True, title='\n\nConfusion Matrix\n')
        self.printer.log_to_file(self.net, classification_report(y_test, predicted), title='\n\nReport\n')
        
        #save a plot of the validation and training losses
        if self.plot_loss:
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
'''
