import lasagne
from lasagne.updates import nesterov_momentum
from lasagne import layers
from my_net import MyNeuralNet
from sklearn.preprocessing import StandardScaler
import FlipBatchIterator as flip
from old_load import RoofLoader
import pdb
from old_print_log import PrintLogSave, SaveLayerInfo
from sklearn.metrics import classification_report, confusion_matrix
IMG_SIZE = 40
CROP_SIZE = 32

#Using FlipBatchIterator we rotate, flip and crop the input for data augmentation purposes
#same call as to NeuralNet, but with a StandardScaler parameter also
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
    input_shape=(None, 3, CROP_SIZE, CROP_SIZE),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=3,
    
    output_nonlinearity=lasagne.nonlinearities.softmax,
    #preproc_scaler = StandardScaler(), 
    preproc_scaler = None,

    update_learning_rate=0.01,
    update_momentum=0.9,
    
    #printing
    net_name='conv3_net_test.out',
    on_epoch_finished=[PrintLogSave()],
    on_training_started=[SaveLayerInfo()],

    batch_iterator_test=flip.CropOnlyBatchIterator(batch_size=128),
    batch_iterator_train=flip.FlipBatchIterator(batch_size=128),
    max_epochs=250,
    verbose=1,
    )

roof_loader = RoofLoader()
X_train, X_test, y_train, y_test, file_names = roof_loader.load()

#rescale X_train
scaler = StandardScaler() 
X_shape = X_train.shape
X_reshaped = X_train.reshape(X_shape[0], X_shape[1]*X_shape[2]*X_shape[3])
X_reshaped = scaler.fit_transform(X_reshaped)
X_train = X_reshaped.reshape(X_shape[0], X_shape[1], X_shape[2], X_shape[3])

#fit the network to X_train
net3.fit(X_train, y_train)

#scale the test set and do predictions
X_valid = X_test
valid_shape = X_valid.shape
X_valid_reshaped = X_valid.reshape(valid_shape[0], valid_shape[1]*valid_shape[2]*valid_shape[3])
X_valid_reshaped = scaler.transform(X_valid_reshaped)
X_valid = X_valid_reshaped.reshape(valid_shape[0], valid_shape[1], valid_shape[2], valid_shape[3])
pdb.set_trace()
predicted = net3.predict(X_valid)

print confusion_matrix(y_test, predicted)
print classification_report(y_test, predicted)

# trained model so that we can load it back later:
import cPickle as pickle
with open('net3.pickle', 'wb') as f:
    pickle.dump(net3, f, -1)
