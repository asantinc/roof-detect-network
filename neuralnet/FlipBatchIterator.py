import sys
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
import numpy as np
from nolearn.lasagne.base import NeuralNet, _sldict, BatchIterator
import scipy.ndimage.interpolation

CROP_SIZE = 32
IMG_SIZE = 40

'''
class BasicBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        temp_Xb = np.zeros((Xb.shape[0], Xb.shape[1], CROP_SIZE, CROP_SIZE))
        xmin = ymin = (IMG_SIZE-CROP_SIZE)/2
        temp_Xb[:,:,:,:] = Xb[:, :, xmin:(xmin+CROP_SIZE), ymin:(ymin+CROP_SIZE)]
        return temp_Xb, yb
'''


class CropOnlyBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        temp_Xb = np.zeros((Xb.shape[0], Xb.shape[1], CROP_SIZE, CROP_SIZE))
        xmin = ymin = (IMG_SIZE-CROP_SIZE)/2
        temp_Xb[:,:,:,:] = Xb[:, :, xmin:(xmin+CROP_SIZE), ymin:(ymin+CROP_SIZE)]
        return temp_Xb, yb


class FlipBatchIterator(BatchIterator):
    '''Subclass of batchiterator that performs data augmentation on the data batches before to feed them into the lasagne NeuralNet 
    '''
    def transform(self, Xb, yb):
        self.Xb = Xb
        self.Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        self.random_flip()
        self.random_rotation(10.)
        self.random_crop()
        return self.Xb, yb

    def random_rotation(self, ang, fill_mode="nearest", cval=0.):
	    angle = np.random.uniform(-ang, ang)
	    self.Xb = scipy.ndimage.interpolation.rotate(self.Xb, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)

    def random_crop(self):
        #Extract 32 by 32 patches from 40 by 40 patches, rotate them randomly
        temp_Xb = np.zeros((self.Xb.shape[0],self.Xb.shape[1], CROP_SIZE, CROP_SIZE))
        margin = IMG_SIZE-CROP_SIZE
        for img in range(self.Xb.shape[0]):
            xmin = np.random.randint(0, margin)
            ymin = np.random.randint(0, margin)
            temp_Xb[img, :,:,:] = self.Xb[img, :, xmin:(xmin+CROP_SIZE), ymin:(ymin+CROP_SIZE)]
        self.Xb = temp_Xb

    def random_flip(self):
        # Flip half of the images in this batch at random:
        bs = self.Xb.shape[0]
        indices_hor = np.random.choice(bs, bs / 2, replace=False)
        indices_vert =  np.random.choice(bs, bs / 2, replace=False)
        self.Xb[indices_hor] = self.Xb[indices_hor, :, :, ::-1]
        self.Xb[indices_vert] = self.Xb[indices_vert, :, ::-1, :]

