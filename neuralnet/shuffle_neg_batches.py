import utils
import numpy as np
import os
from random import shuffle
import pdb
import subprocess


###########
# Shuffle the negative training data from the sliding window
###########

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def save_patches(patch_names, patch_paths, path):
    for p, (patch_name, patch_path) in enumerate(zip(patch_names, patch_paths)):
        new_path = '{}batch{}_shuffled/'.format(path,int(p/5000))       
        if p%5000 == 0:
            utils.mkdir(new_path)
        subprocess.check_call('mv {} {}{}'.format(patch_path, new_path, patch_name), shell=True) 
       

if __name__ == '__main__':
    #training_path = utils.get_path(in_or_out=utils.IN, data_fold=utils.TRAINING, neural=True)+'falsepos/'
    training_path = '../slide_training_data_neural/scale1.3_minSize50-50_windowSize15-15_stepSize30_dividedSizes/falsepos/'
    print training_path
    metal_names = list()
    metal_paths = list()
    thatch_names = list()
    thatch_paths = list()
    for batch in os.listdir(training_path):
        print 'BATCH:', batch
        for file in os.listdir(training_path+batch):
            file_path = training_path+batch+'/'+file
            if file.startswith('metal'):
                metal_names.append(file)
                metal_paths.append(file_path)
            elif file.startswith('thatch'):
                thatch_names.append(file)
                thatch_paths.append(file_path)
    metal_names = np.array(metal_names, dtype=object)
    thatch_names = np.array(thatch_names, dtype=object)
    metal_paths = np.array(metal_paths, dtype=object)
    thatch_paths = np.array(thatch_paths, dtype=object)

    metal_names, metal_paths = shuffle_in_unison(metal_names, metal_paths)
    thatch_names, thatch_paths = shuffle_in_unison(thatch_names, thatch_paths)

    pdb.set_trace()
    save_patches(metal_names, metal_paths, training_path)
    save_patches(thatch_names, thatch_paths, training_path)

