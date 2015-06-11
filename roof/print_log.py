from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.objectives import Objective

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

import load
import sys
import pdb
import lasagne
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

from nolearn.lasagne.handlers import PrintLog 

class PrintLogSave(PrintLog):
    
    def __call__(self, nn, train_history):
        file = open(nn.net_name, 'a')
        file.write(self.table(nn, train_history))
        file.close()

    def table():


