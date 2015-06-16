from collections import OrderedDict
from datetime import datetime
from functools import reduce
import numpy as np
import operator
import sys
import pdb
from tabulate import tabulate

import lasagne
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
from nolearn.lasagne.handlers import PrintLog, PrintLayerInfo 
from nolearn.lasagne.util import is_conv2d

#my modules
import load
from sklearn.metrics import classification_report, confusion_matrix

#Constants
OUT_PATH = "../output/" 
FTRAIN = '../data/train/'
FTRAIN_LABEL = '../data/labels.csv'
FTEST = '../data/test/'
IMG_SIZE = 40
CROP_SIZE = 32


class Experiment(object):
    def __init__(self, net=None, data_augmentation=True, display_mistakes=False, 
                test_percent=.10, scaler=True, preloaded=True, printer=None):
        self.net=net
        self.data_augmentation=data_augmentation
        self.test_percent=test_percent
        self.scaler=scaler
        self.preloaded=preloaded
        self.printer=printer
        self.display_mistakes=display_mistakes

    def __str__(self):
        out_list = list()
        for key, value in self.__dict__.items():
            out_list.append(str(key)+': '+str(value))
        out_list.append('\n')
        return '\n'.join(out_list)          

    def run(self):
        #save settings to file
        self.printer.log_to_file(self.net, self.__str__(), overwrite=True)

        #load data
        roof_loader = load.RoofLoader()
        X_train, X_test, y_train, y_test, file_names = roof_loader.load(test_percent=self.test_percent)

        #rescale X_train and X_test
        if self.scaler:
            scaler = load.DataScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform2(X_test)

        #only train the network if we choose not to preload weights
        if self.preloaded:
            self.net.load_params_from('saved_weights/'+self.net.net_name+'.pickle')
        else:
            #fit the network to X_train
            self.net.fit(X_train, y_train)
            self.net.save_weights()

        #find predictions for test set
        predicted = self.net.predict(X_test)

        #print evaluation
        self.printer.log_to_file(self.net, confusion_matrix(y_test, predicted), binary=True, title='\n\nConfusion Matrix\n')
        self.printer.log_to_file(self.net, classification_report(y_test, predicted), title='\n\nReport\n')

        #display mistakes
        if self.display_mistakes: 
            mistakes = np.array([True if y_test[i]-predicted[i] != 0 else False for i in range(len(y_test))])
            mistaken_imgs = X_test[mistakes]
            mistaken_imgs = scaler.inverse_transform(mistaken_imgs)
            roof_loader.display_images(mistaken_imgs, labels=y_test[mistakes], indeces=range(len(mistaken_imgs)))


class PrintLogSave(PrintLog):
    def __call__(self, nn, train_history): 
        file = open(OUT_PATH+nn.net_name, 'a')
        file.write(self.table(nn, train_history))
        file.close()

    def table(self, nn, train_history):
        info = train_history[-1]
        return str(info['epoch'])+'\t'+ str(info['train_loss'])+ '\t'+str(info['valid_loss'])+'\t'+str( info['train_loss'] / info['valid_loss'])+'\t'+str(info['valid_accuracy'])+'\n'
            
    def log_to_file(self, nn, log, overwrite=False, binary=False, title=''):
        write_type = 'w' if overwrite else 'a'
        file = open(OUT_PATH+nn.net_name, write_type)
        file.write(title)
        if binary:
            print >> file, log
        else:
            file.write(log)
        file.close()


class SaveLayerInfo(PrintLayerInfo):
    def __call__(self, nn, train_history):
        file = open(OUT_PATH+nn.net_name, 'a')
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

        

