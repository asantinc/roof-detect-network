from collections import OrderedDict
from datetime import datetime
from functools import reduce
import operator
import sys
import pdb
from tabulate import tabulate

import lasagne

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

from nolearn.lasagne.handlers import PrintLog, PrintLayerInfo 
from nolearn.lasagne.util import is_conv2d

#Constants
OUT_PATH = "../output/" 
FTRAIN = '../data/train/'
FTRAIN_LABEL = '../data/labels.csv'
FTEST = '../data/test/'
IMG_SIZE = 40
CROP_SIZE = 32

class PrintLogSave(PrintLog):
    def __call__(self, nn, train_history): 
        file = open(OUT_PATH+nn.net_name, 'a')
        file.write(self.table(nn, train_history))
        file.close()

    def table(self, nn, train_history):
        info = train_history[-1]
        return str(info['epoch'])+'\t'+ str(info['train_loss'])+ '\t'+str(info['valid_loss'])+'\t'+str( info['train_loss'] / info['valid_loss'])+'\t'+str(info['valid_accuracy'])+'\n'
            
    def log_to_file(self, nn, log, overwrite=False):
        write_type = 'w' if overwrite else 'a'
        file = open(OUT_PATH+nn.net_name, write_type)
        file.write(log)
        file.close()


class ExperimentSettings(object):
    def __str__(self):
        out_list = list()
        for key, value in self.__dict__.items():
            out_list.append(str(key)+': '+str(value))
        out_list.append('\n')
        return '\n'.join(out_list)          


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

        

