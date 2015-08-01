import sys
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
import numpy as np
from nolearn.lasagne.base import NeuralNet, _sldict, BatchIterator
import scipy.ndimage.interpolation

import utils
from data_augmented import Augmenter

CROP_SIZE = 32
IMG_SIZE = 40



class ResizeBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        temp_Xb = np.empty((Xb.shape[0], Xb.shape[1], CROP_SIZE, CROP_SIZE))
        for i, x in enumerate(Xb):
            x_transposed = x.transpose(1,2,0) 
            x_resized = utils.resize_rgb(x_transposed, w=CROP_SIZE, h=CROP_SIZE)
            x_new = x_resized.transpose(2,0,1)
            temp_Xb[i, :, :, :]  = x_new
        return temp_Xb, yb


class FlipBatchIterator(BatchIterator):
    '''Subclass of batchiterator that performs data augmentation on the data batches before to feed them into the lasagne NeuralNet 
    '''
    def transform(self, Xb, yb):
        Xb = Xb
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        for i, x in self.Xb:
            patch = utils.neural_to_cv2(x)
            patch = Augmenter.random_flip(patch) 
            patch = Augmenter.random_rotation(patch, 10)
            patch = Augmenter.random_crop(patch, (CROP_SIZE, CROP_SIZE))
            #patch = Augmenter.random_illum(patch)
            Xb[i, :, :,: ] = utils.neural_to_cv2(patch)
        return Xb, yb

