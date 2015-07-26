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

import get_data
from get_data import DataLoader
from augmentation import DataAugmentation
import utils

'''
ViolaDataSetup is used to:
    - setup the training patches for viola
    - produce the corresponding .dat and .vec files
The .vec files are needed to train a cascade from the ViolaTrainer class
'''

class NeuralDataLoad(object):
    def load_training_imgs():
        #load thatch from viola TP
        #load thatch from viola training

        in_path_source = utils.get_path(in_or_out=utils.IN, data_fold=utils.TRAINING)    
        for roof_type in ['metal', 'thatch']:
            normal_patches[roof_type] = get_neural_patches(roof_type, in_path_source)
        viola_pos_folder = 'combo4_min_neighbors3_scale1.08_groupNone_removeOffTrue_rotateTrue/True/'
        :tabedit 

        #load metal 

        #load negative examples from: 1. Viola FP; 2. uninhabited images

    def get_neural_patches(roof_type, in_path):
        in_path_source = utils.get_path(in_or_out=utils.IN, data_fold=utils.TRAINING)    
        DataLoader.get_polygons(roof_type=roof_type,  

    @staticmethod
    def setup_augmented_patches():
        '''
        No division between different roof sizes: if a roof has a size that is off, we resize it
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''
        in_path = utils.get_path(in_or_out=utils.IN, data_fold=utils.TRAINING)
        out_path = utils.get_path(neural=True, in_or_out=utils.IN, data_fold=utils.TRAINING)

        img_names_list = [img_name for img_name in os.listdir(in_path) if img_name.endswith('.jpg')]

        for roof_type in ['metal', 'thatch']:
            for img_id, img_name in enumerate(img_names_list):

                print 'Processing image: {0}'.format(img_name)
                img_path = in_path+img_name

                polygon_list = DataLoader.get_polygons(roof_type=roof_type, xml_name=img_name[:-3]+'xml', xml_path=in_path)
                roof_patches = DataLoader.extract_patches(polygon_list, img_path=img_path, grayscale=True)

                for roof_id, roof_img in enumerate(roof_patches):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)
                       
                    #if it's vertical, make it lie down
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    
                    #write basic positive example to the right folder
                    general_path = '{0}{1}_{2}_{3}'.format(out_path, roof_type, img_name[:-4], roof_id)

                    #calculate and write the augmented images 
                    for i in range(4):
                        roof_img_cp = np.copy(roof_img)

                        if i == 1:
                            roof_img_cp = cv2.flip(roof_img_cp,flipCode=0)
                        elif i == 2:
                            roof_img_cp = cv2.flip(roof_img_cp,flipCode=1)
                        elif i==3:
                            roof_img_cp = cv2.flip(roof_img_cp,flipCode=-1)

                        write_to_path = '{0}_flip{1}.jpg'.format(general_path, i)
                        cv2.imwrite(write_to_path, roof_img_cp)


########################################
## FINAL CODE USED BELOW
########################################

def main(data_reset=False):


if __name__ == '__main__':
    main(data_reset=False)


