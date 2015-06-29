from lasagne import layers
from lasagne.updates import nesterov_momentum
import load
import sys
import pdb
import lasagne
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 40
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

#pdb.set_trace()
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from my_net import MyNeuralNet
from print_log import PrintLogSave, SaveLayerInfo


net2 = MyNeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 3*IMG_SIZE*IMG_SIZE), 
    output_num_units=2,
    
    #printing
    net_name='logistic_roofs_only',
    on_epoch_finished=[PrintLogSave()],
    on_training_started=[SaveLayerInfo()],

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.001,
    update_momentum=0.9,

    output_nonlinearity=lasagne.nonlinearities.softmax,
    max_epochs=500,  # we want to train this many epochs
    verbose=1,
    )

X, y, _ = load.load(roof_only=True)
print np.bincount(y)
pdb.set_trace()
X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
net2.fit(X, y)


