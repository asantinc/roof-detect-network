
from collections import OrderedDict
from datetime import datetime
from functools import reduce
import operator

from tabulate import tabulate

import pdb
import lasagne

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

from nolearn.lasagne.handlers import PrintLog 

class PrintLogSave(PrintLog):
    
    def __call__(self, nn, train_history):
        file = open(nn.net_name, 'a')
        file.write(self.table(nn, train_history))
        file.close()

    def table(self, nn, train_history):
        info = train_history[-1]

        return str(info['epoch'])+'\t'+ "{}{:.5f}{}".format(info['train_loss']))+ '\t'+"{}{:.5f}{}".format(info['valid_loss']))+'\t'+
            str( info['train_loss'] / info['valid_loss'])+'\t'+st(info['valid_accuracy'])
            
    

