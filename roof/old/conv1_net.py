import lasagne
from lasagne.updates import nesterov_momentum
from lasagne import layers
#from nolearn.lasagne import NeuralNet
from my_net import MyNeuralNet
from sklearn.preprocessing import StandardScaler
import load
import pdb
from print_log import PrintLogSave, SaveLayerInfo

IMG_SIZE = 40

#StandardScaler normalizes data
#No data augmentation performed
net2 = MyNeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, IMG_SIZE, IMG_SIZE), 
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    
    output_num_units=3,
    output_nonlinearity=lasagne.nonlinearities.softmax,
    preproc_scaler = StandardScaler(), 
    
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=1000,
    verbose=1,

    #printing
    net_name='conv1_net.out',
    on_epoch_finished=[PrintLogSave()],
    on_training_started=[SaveLayerInfo()],
    )

X, y, _ = load.load()  # load 2-d data
net2.fit(X, y)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)
