from lasagne import layers
from lasagne.updates import nesterov_momentum
import load
import sys
import pdb
import lasagne
import numpy as np

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

#pdb.set_trace()
from nolearn.lasagne import NeuralNet
net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 4800),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    #output_nonlinearity=None,  # output layer uses identity function
    output_num_units=3,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    #regression=True,  # flag to indicate we're dealing with regression problem
    output_nonlinearity=lasagne.nonlinearities.softmax,
    max_epochs=50,  # we want to train this many epochs
    verbose=1,
    )

X, y = load.load()
#pdb.set_trace()
net1.fit(X, y)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)

pyplot.show()
