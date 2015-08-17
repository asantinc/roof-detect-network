import os
import subprocess
import pdb
import math
import random
import getopt
import sys
import csv
import numpy as np
import cv2
from scipy import misc, ndimage #load images
from collections import defaultdict
import sklearn.utils

import get_data
from get_data import DataLoader
from augmentation import DataAugmentation
import utils
from FlipBatchIterator import FlipBatchIterator as flip

'''
ViolaDataSetup is used to:
    - setup the training patches for viola
    - produce the corresponding .dat and .vec files
The .vec files are needed to train a cascade from the ViolaTrainer class
'''


class NeuralDataLoad(object):
    def __init__(self, data_path = None, full_dataset=False, method='viola'):
        self.full_dataset=full_dataset
        self.method = method
        #if full_dataset == False and method=='viola':
        path = utils.get_path(data_fold=utils.TRAINING, neural=True, in_or_out = utils.IN, method=method, full_dataset=full_dataset)
        #elif full_dataset == False:
        #    path = '../slide_data_training_neural/'    
        #else:
            #this data is now in your harddrive!!!!
            #path = '../training_data_full_dataset_neural/'
        self.background_FP_viola_path = '{}{}/falsepos/'.format(path, data_path)
        self.thatch_metal_TP_viola_path = '{}{}/truepos/'.format(path, data_path)
        print self.background_FP_viola_path
        print self.thatch_metal_TP_viola_path


    def load_data(self, non_roofs=None, roof_type=None, starting_batch=0):
        '''
        Parameters:
        ----------
        non_roofs: float
            Determines what proportion of non_roofs should be added to the dataset
        roof_type: string
            If roof_type equals 'metal' or 'thatch' we only load patches for 
            that type of roof. Otherwise, we load both types
        starting_batch: int
            when doing an ensemble, we specify which batch we want to start picking up data from
        '''
        assert roof_type=='metal' or roof_type=='thatch' or roof_type=='Both'
        #First get the positive patches
        self.ground_truth_metal_thatch = DataLoader.get_all_patches_folder(merge_imgs=True, full_dataset=self.full_dataset)
        self.viola_metal_thatch = self.get_viola_positive_patches(self.thatch_metal_TP_viola_path)
        total_length = self.count_patches_helper(non_roofs)

        self.X = np.empty((total_length, 3, utils.PATCH_W, utils.PATCH_H), dtype='float32')
        self.y = np.empty((total_length), dtype='int32')
        self.failed_patches = 0

        #process the metal and thatch
        self.roof_types = [roof_type] if roof_type!='Both' else utils.ROOF_TYPES
        index = 0 
        for roof_type in self.roof_types:
            if len(self.roof_types) > 1:
                label = utils.ROOF_LABEL[roof_type]
            else:
                label = 1
            for data_source in [self.ground_truth_metal_thatch[roof_type], self.viola_metal_thatch[roof_type]]: 
                for patch in data_source:
                    index = self.process_patch(patch, label, index)                    

        

        #limit the number of background patches
        self.non_roof_limit = (non_roofs*index) + index
        #BACKGROUND
        if self.method == 'slide': #self.full_dataset: #if we want the full dataset (sliding window only( then we have to access it in batches)
            self.viola_background = self.get_background_patches_from_batches(self.background_FP_viola_path, roof_type, starting_batch=starting_batch)
        else:
            if roof_type != 'Both':
                print 'Will load {0} data only'.format(roof_type)
                self.viola_background = self.get_viola_background_patches(self.background_FP_viola_path, roof_type=roof_type)
            else:
                self.viola_background = self.get_viola_background_patches(self.background_FP_viola_path)
        label = utils.ROOF_LABEL['background']

        for patch in self.viola_background:
            if index > self.non_roof_limit: #if we have obtained enough non_roofs, break
                break
            index = self.process_patch(patch, label, index)
        #here we can add more random negative patches if needed

        #remove the end if index<len(X) -- some patches failed to load
        self.X = self.X[:index, :,:,:]
        #self.X = self.X.astype(np.float32)
        self.y = self.y[:index]
        self.y = self.y.astype(np.int32)
        

        print np.bincount(self.y)
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y, random_state=42)  # shuffle train data    

        #utils.debug_data(self.X, self.y, index, roof_type, flip(batch_size=128))
        return self.X, self.y


    def get_viola_positive_patches(self, path):
        #look at the viola folder, get all of the jpg.s depending on what roof_type the image name contains
        viola_metal_thatch = defaultdict(list)
        for file in os.listdir(self.thatch_metal_TP_viola_path):
            if file.endswith('.jpg'):
                patch = cv2.imread(path+file)
                if 'metal' in file.lower():
                    viola_metal_thatch['metal'].append(patch) 
                elif 'thatch' in file.lower():
                    viola_metal_thatch['thatch'].append(patch)
        return viola_metal_thatch


    def get_viola_background_patches(self, path, roof_type=None): 
        background_patches = list()
        for file in os.listdir(path):
            if file.endswith('.jpg'):
                if roof_type is None:
                    if 'background' in file.lower():
                        patch = cv2.imread(path+file)
                        background_patches.append(patch) 
                else:
                    if roof_type in file.lower():
                        patch = cv2.imread(path+file)
                        background_patches.append(patch) 
        np.random.shuffle(np.array(background_patches))
        return background_patches 


    def get_background_patches_from_batches(self, FP_path, roof_type, starting_batch=0):  
        '''For the ensembles, we need to spify the starting_batch
        '''
        #get the batches from the folder
        total_patches = 0
        background_patches = list()
        batches = [b for b in os.listdir(FP_path) if b.endswith('_shuffled')]
        if starting_batch>0 and starting_batch < len(batches):
            batch_start = batches[starting_batch:]
            batch_end = batches[:starting_batch]
            batches = batch_start+batch_end

        #get the files from each batch
        for batch in batches:
            path = FP_path+batch+'/'
            for file in os.listdir(path):
                if file.startswith(roof_type):
                    total_patches += 1
                    patch  = cv2.imread(path+file)
                    background_patches.append(patch)
                    #stop when we've grabbed enough
                    if total_patches>self.non_roof_limit:
                        break
            if total_patches>self.non_roof_limit:
                break
        return background_patches


    def process_patch(self, patch, label, index):
        try:
            x = utils.cv2_to_neural(patch)
            x.shape = (1, x.shape[0], x.shape[1], x.shape[2])
            self.X[index, :, :, :] = x
        except ValueError, e:
            print e
            self.failed_patches += 1
            print 'fail:'+ str(failures)
        else:
            self.y[index] = label
            index += 1
        return index 


    def count_patches_helper(self, non_roofs):
        total_length = 0 
        for roof_type in utils.ROOF_TYPES:
            total_length += len(self.viola_metal_thatch[roof_type])
            total_length += len(self.ground_truth_metal_thatch[roof_type])
        #this final total includes the number of non_roofs that we expect to load
        total_length += total_length*non_roofs
        return total_length

    @staticmethod
    def save_img(x,y,i, other_text=''):
        x = x*255
        x = x.transpose(1,2,0)
        cv2.imwrite('check_load_neural/{2}_img{0}_label{1}.jpg'.format(i, y[i], other_text), x)



def main(data_reset=False):
    X, y = NeuralDataLoad().load_data()
    bins = np.bincount(y)
    print bins 
    print X.shape
    print y.shape
    for i in range(X.shape[0]):
        x = X[i, :, :,:]
        NeuralDataLoad.save_img(x,y, i)

if __name__ == '__main__':
    main()


