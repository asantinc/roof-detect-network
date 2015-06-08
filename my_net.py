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

IMG_SIZE = 40
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

from nolearn.lasagne.base import NeuralNet, _sldict, BatchIterator

class MyNeuralNet(NeuralNet):
	'''
	Subclass of NeuralNet that incorporates scaling of the data
	'''
	def __init__(
	        self,
	        layers,
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
		    more_params=more_params,
		    verbose=verbose,
		    **kwargs)
		self.regression = regression
		self.batch_iterator_test = batch_iterator_test
		self.preproc_scaler = preproc_scaler

	def predict_proba(self, X):
	    if self.preproc_scaler is not None:
	        X = self.preproc_scaler.transform(X)
	    probas = []
	    for Xb, yb in self.batch_iterator_test(X):
	        probas.append(NeuralNet.apply_batch_func(self.predict_iter_, Xb))
		return np.vstack(probas)

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
            pdb.set_trace()
            X_train = self.preproc_scaler.fit_transform(X_train)
            X_valid = self.preproc_scaler.transform(X_valid)

	    return X_train, X_valid, y_train, y_valid




if __name__ == "__main__":
    net = MyNeuralNet()
