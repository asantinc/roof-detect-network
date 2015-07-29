import load
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import pdb

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.objectives import Objective

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

from nolearn.lasagne.base import NeuralNet, _sldict, BatchIterator
import utils

class MyNeuralNet(NeuralNet):
	'''
	Subclass of NeuralNet that incorporates scaling of the data
	'''
	def __init__(
	        self,
	        layers,
            num_layers=0,
	        update=nesterov_momentum,
	        loss=None,
            	objective=Objective,
            	objective_loss_function=None,
	        batch_iterator_train=BatchIterator(batch_size=128),
	        batch_iterator_test=BatchIterator(batch_size=128),
	        regression=False,
	        max_epochs=100,
	        eval_size=0.2,
            	custom_score=None,
	        X_tensor_type=None,
	        y_tensor_type=None,
	        use_label_encoder=False,
	        on_epoch_finished=None,
            on_training_started=None,
	        on_training_finished=None,
	        preproc_scaler=None,
	        more_params=None,
	        verbose=0,
           	net_name='no_name',
	        **kwargs):
		NeuralNet.__init__(
			self,
		    layers,
		    update=update,
		    loss=loss,
		    objective=objective,
		    objective_loss_function=objective_loss_function,
		    batch_iterator_train=batch_iterator_train,
		    batch_iterator_test=batch_iterator_test,
		    regression=regression,
		    max_epochs=max_epochs,
		    eval_size=eval_size,
		    custom_score=custom_score,
		    X_tensor_type=X_tensor_type,
		    y_tensor_type=y_tensor_type,
		    use_label_encoder=use_label_encoder,
		    on_epoch_finished=on_epoch_finished,
		    on_training_finished=on_training_finished,
            on_training_started=on_training_started,
		    more_params=more_params,
		    verbose=verbose,
		    **kwargs)
		self.net_name = net_name
		self.regression = regression
		self.batch_iterator_test = batch_iterator_test
        	self.preproc_scaler = preproc_scaler
                self.num_layers=num_layers


	def train_test_split(self, X, y, eval_size):
	    if eval_size:
	        if self.regression:
	            kf = KFold(y.shape[0], round(1. / eval_size))
	        else:
	            kf = StratifiedKFold(y, round(1. / eval_size))

	        train_indices, valid_indices = next(iter(kf))
	        X_train, y_train = X[train_indices], y[train_indices]
	        X_valid, y_valid = X[valid_indices], y[valid_indices]
	    else:
	        X_train, y_train = X, y
	        X_valid, y_valid = _sldict(X, slice(len(X), None)), y[len(y):]
	    if self.preproc_scaler is not None:
                train_shape = X_train.shape
                X_train_reshaped = X_train.reshape(train_shape[0], train_shape[1]*train_shape[2]*train_shape[3]) 
                X_train_reshaped = self.preproc_scaler.fit_transform(X_train_reshaped)
                X_train = X_train_reshaped.reshape(train_shape[0], train_shape[1], train_shape[2], train_shape[3])
                
                valid_shape = X_valid.shape 
                X_valid_reshaped = X_valid.reshape(valid_shape[0], valid_shape[1]*valid_shape[2]*valid_shape[3])
                X_valid_reshaped = self.preproc_scaler.transform(X_valid_reshaped)
                X_valid = X_valid_reshaped.reshape(valid_shape[0], valid_shape[1], valid_shape[2], valid_shape[3])

	    return X_train, X_valid, y_train, y_valid
   

        def save_weights(self):
            ''' Saves weigts of model so they can be loaded back later:
            '''
            path = utils.get_path(params=True, neural_weights=True) #'../parameters/saved_weights/'
            with ('{0}_{1}_accur{0:.4f}.pickle'.format(path, self.net_name, self.valid_accuracy), 'wb') as f:
                pickle.dump(self, f, -1)
        
        
        def save_loss(self):
            '''Save the plot of the training and validation loss
            '''
            train_loss = [row['train_loss'] for row in self.train_history_]
            valid_loss = [row['valid_loss'] for row in self.train_history_]
            plt.plot(train_loss, label='train loss')
            plt.plot(valid_loss, label='valid loss')
            plt.legend(loc='best')
            path = utils.get_path(neural=True, in_or_out=utils.OUT, data_fold=utils.TRAIN) 
            plt.savefig(path+self.net_name+'_loss.png')
        

        @staticmethod
        def produce_layers(num_layers=1):
            assert num_layers<=5
            assert num_layers>=0
            if num_layers==0:
                net_layers=[
                    ('input', layers.InputLayer),
                    ('output', layers.DenseLayer),
                    ]
                params_dict = {'conv1_num_filters':32, 'conv1_filter_size':(3, 3), 'pool1_pool_size':(2, 2)}

            elif num_layers==1:
                net_layers=[
                    ('input', layers.InputLayer),
                    ('conv1', layers.Conv2DLayer),
                    ('pool1', layers.MaxPool2DLayer),
                    ('output', layers.DenseLayer),
                    ]
                params_dict = {'conv1_num_filters':32, 'conv1_filter_size':(3, 3), 'pool1_pool_size':(2, 2),
                        'conv2_num_filters':64, 'conv2_filter_size':(2, 2), 'pool2_pool_size':(2, 2)}

            elif num_layers==2:
                net_layers=[
                    ('input', layers.InputLayer),
                    ('conv1', layers.Conv2DLayer),
                    ('pool1', layers.MaxPool2DLayer),
                    ('conv2', layers.Conv2DLayer),
                    ('pool2', layers.MaxPool2DLayer),
                    ('output', layers.DenseLayer),
                    ]
                params_dict = {'conv1_num_filters':32, 'conv1_filter_size':(3, 3), 'pool1_pool_size':(2, 2),
                        'conv2_num_filters':64, 'conv2_filter_size':(2, 2), 'pool2_pool_size':(2, 2)}

            elif num_layers==3:
                net_layers=[
                    ('input', layers.InputLayer),
                    ('conv1', layers.Conv2DLayer),
                    ('pool1', layers.MaxPool2DLayer),
                    ('conv2', layers.Conv2DLayer),
                    ('pool2', layers.MaxPool2DLayer),
                    ('conv3', layers.Conv2DLayer),
                    ('pool3', layers.MaxPool2DLayer),
                    ('output', layers.DenseLayer),
                    ]
                params_dict = {'conv1_num_filters':32, 'conv1_filter_size':(3, 3), 'pool1_pool_size':(2, 2),
                        'conv2_num_filters':64, 'conv2_filter_size':(2, 2), 'pool2_pool_size':(2, 2),
                        'conv3_num_filters':128, 'conv3_filter_size':(2, 2), 'pool3_pool_size':(2, 2)}
            elif num_layers==4:
                net_layers=[
                    ('input', layers.InputLayer),
                    ('conv1', layers.Conv2DLayer),
                    ('pool1', layers.MaxPool2DLayer),
                    ('conv2', layers.Conv2DLayer),
                    ('pool2', layers.MaxPool2DLayer),
                    ('conv3', layers.Conv2DLayer),
                    ('pool3', layers.MaxPool2DLayer),
                    ('hidden4', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ]
                params_dict = {'conv1_num_filters':32, 'conv1_filter_size':(3, 3), 'pool1_pool_size':(2, 2),
                        'conv2_num_filters':64, 'conv2_filter_size':(2, 2), 'pool2_pool_size':(2, 2),
                        'conv3_num_filters':128, 'conv3_filter_size':(2, 2), 'pool3_pool_size':(2, 2),
                        'hidden4_num_units':500}
            elif num_layers==5:
                net_layers=[
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
                    ]
                params_dict = {'conv1_num_filters':32, 'conv1_filter_size':(3, 3), 'pool1_pool_size':(2, 2),
                        'conv2_num_filters':64, 'conv2_filter_size':(2, 2), 'pool2_pool_size':(2, 2),
                        'conv3_num_filters':128, 'conv3_filter_size':(2, 2), 'pool3_pool_size':(2, 2),
                        'hidden4_num_units':500, 'hidden5_num_units':500}
            return net_layers, params_dict 


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print "Early stopping"
            print("Best valid loss was {:.6f} at epoch {}.".format(
                                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()



class SaveBestWeights(object):
    def __init__(self, patience=100):
        self.best_valid = np.inf
        self.weight_path = utils.get_path(params=True, neural_weights=True)

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_weights = nn.get_all_params_values()
            
            weight_name = '{0}{1}.pickle'.format(self.weight_path, nn.net_name, self.best_valid, current_epoch) 
            with open(weight_name, 'wb') as f:
                pickle.dump(nn, f, -1)                 

if __name__ == "__main__":
    net = MyNeuralNet()

