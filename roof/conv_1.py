import pdb

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

#my modules
from my_net import MyNeuralNet
import FlipBatchIterator as flip
import load
import experiment_settings as settings
from experiment_settings import PrintLogSave, SaveLayerInfo, ExperimentSettings


'''Convolutional neural network with data augmentation (Single convolutional layer)

Using the subclass of nolearn's neural network, we can pass a FlipBatchIterator 
as the batch iterator. This class rotates, flip and crops the inputs for data augmentation purposes.
The test data is also cropped before classifying it.

In this case we have chosen to standardize the training data before inputing it into the
network. The validation set is extracted automatically by the network from this training set.
As a result, the validation metrics produced are corrupted, so to measure performance we can only rely on the
testing metrics.
'''

#experiment settings
exper = ExperimentSettings()
exper.epochs = 250
exper.net_name = 'conv_1'
exper.data_augmentation = True 
exper.test_percent = .10
exper.scaler = 'StandardScaler'
       
#declare the conv. net
printer = PrintLogSave()
net = MyNeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, settings.CROP_SIZE, settings.CROP_SIZE),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    output_num_units=3,
    
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,
    
    #printing
    net_name=exper.net_name,
    on_epoch_finished=[printer],
    on_training_started=[SaveLayerInfo()],

    #data augmentation
    batch_iterator_test=flip.CropOnlyBatchIterator(batch_size=128),
    batch_iterator_train=flip.FlipBatchIterator(batch_size=128),
    
    max_epochs=exper.epochs,
    verbose=1,
    )
#save settings to file
printer.log_to_file(net, exper.__str__(), overwrite=True)

#load data
roof_loader = load.RoofLoader()
X_train, X_test, y_train, y_test, file_names = roof_loader.load(test_percent=exper.test_percent)

#rescale X_train and X_test
scaler = load.DataScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform2(X_test)

#fit the network to X_train
net.fit(X_train, y_train)

#find predictions for test set
predicted = net.predict(X_test)

#print evaluation
printer.log_to_file(net, confusion_matrix(y_test, predicted), binary=True, title='\n\nConfusion Matrix\n')
printer.log_to_file(net, classification_report(y_test, predicted), title='\n\nReport\n')

