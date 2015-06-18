import pdb
import sys
import getopt

import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.preprocessing import StandardScaler

#my modules
from my_net import MyNeuralNet
import FlipBatchIterator as flip
import load
import experiment_settings as settings
from experiment_settings import PrintLogSave, SaveLayerInfo, Experiment


'''Convolutional neural network with data augmentation

Using the subclass of nolearn's neural network, we can pass a FlipBatchIterator 
as the batch iterator. This class rotates, flip and crops the inputs for data augmentation purposes.
The test data is also cropped before classifying it.

In this case we have chosen to standardize the training data before inputing it into the
network. The validation set is extracted automatically by the network from this training set.
As a result, the validation metrics produced are corrupted, so to measure performance we can only rely on the
testing metrics.
'''


def convolution(
        epochs=250, 
        log=True, 
        plot_loss=False, 
        net_name=None, 
        test_percent=.20, 
        non_roofs=1, 
        preloaded=False, 
        num_layers=1, 
        roofs_only=False):    
    
    printer = PrintLogSave()
    name_percent = str(int(100*test_percent))
    net_name = 'conv'+str(num_layers)+'_nonroofs'+str(non_roofs)+'_test'+name_percent if net_name==None else net_name
    if roofs_only:
        net_name = net_name+'_roofs'
    
    layers = MyNeuralNet.produce_layers(num_layers)    
    
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

    #more experiment settings dependent on number of layers
    if num_layers==5:
        experiment.net.set_params(conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500)
    elif num_layers==4:
        experiment.net.set_params(conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=500)
    elif num_layers==3:
        experiment.net.set_params(conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2))
    elif num_layers==2:
        experiment.net.set_params(conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2))
    elif num_layers==1:
        experiment.net.set_params(conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2))

    experiment.run(log=log, plot_loss=plot_loss) 

def command_line_process(opts):
    test_percent=0.2
    non_roofs=1
    preloaded=False
    num_layers=0 #logistic
    roofs_only=True
    plot=True
    net_name=None
    epoch=250
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
            epoch=int(float(arg))
    return test_percent, non_roofs, preloaded, num_layers, roofs_only, plot, net_name, epoch

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:n:l:p:r:a:e:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)
    test_percent, non_roofs, preloaded, num_layers, roofs_only, plot, net_name, epoch = command_line_process(opts) 
    
    convolution(epochs=epoch, net_name=net_name, test_percent=test_percent, non_roofs=non_roofs, preloaded=preloaded, num_layers=num_layers, roofs_only=roofs_only, plot_loss=plot)

