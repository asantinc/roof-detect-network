import sys
print __name__
sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')
import numpy as np
from nolearn.lasagne.base import NeuralNet, _sldict, BatchIterator
import pdb

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices_hor = np.random.choice(bs, bs / 2, replace=False)
        indices_vert =  np.random.choice(bs, bs / 2, replace=False)
        Xb[indices_hor] = Xb[indices_hor, :, :, ::-1]
        Xb[indices_vert] = Xb[indices_vert, :, ::-1, :]

        return Xb, yb
