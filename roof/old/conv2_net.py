import lasagne
from lasagne.updates import nesterov_momentum
from lasagne import layers
from my_net import MyNeuralNet
from sklearn.preprocessing import StandardScaler
import FlipBatchIterator as flip
import load
import pdb
from print_log import PrintLogSave, SaveLayerInfo

IMG_SIZE = 40

#same call as to NeuralNet, but with a StandardScaler parameter also
#This network uses the FlipBatchIterator to make flips of the input

net3 = MyNeuralNet(
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
    
    #printing
    net_name='conv2_net.out',
    on_epoch_finished=[PrintLogSave()],
    on_training_started=[SaveLayerInfo()],

    batch_iterator_train=flip.FlipBatchIterator(batch_size=128),
    max_epochs=3000,
    verbose=1,
    )

X, y, data_stats = load.load() 
print data_stats
net3.fit(X, y)

# Training for 1000 epochs will take a while.  We'll pickle the
# trained model so that we can load it back later:
import cPickle as pickle
with open('net3.pickle', 'wb') as f:
    pickle.dump(net3, f, -1)
