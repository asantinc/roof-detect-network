import pdb
import sys
import os
import matplotlib.pyplot as plt
import pdb

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

from nolearn.lasagne.visualize import plot_loss
import convolution


def plot_loss():
    for file in os.listdir('saved_weights'):
        if file.endswith(".pickle") and file.startswith('conv'):
            #get the layer number: assuming string format is conv1*.pickle
            num_layers = int(float(file[4:5])) 
            convolution.convolution(preloaded=True, log=False, num_layers=num_layers,plot_loss=True, net_name=file[:-7])             


if __name__ == "__main__":
    plot_loss()

