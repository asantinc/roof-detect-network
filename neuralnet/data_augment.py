import numpy as np
import pdb  
import os #import the image files
import random #get random patches
import pickle
import itertools
import subprocess

import cv2
import utils

from get_data import DataLoader

class Augmenter(object):
    @staticmethod
    def random_flip(img):
        rand_num = random.random()
        if rand_num<0.25:
            return img
        elif rand_num < 0.5:
            flipCode = 0
        elif rand_num < 0.75:
            flipCode = 1
        else:
            flipCode = -1
        img = cv2.flip(img,flipCode=flipCode)
        return img


    @staticmethod
    def random_rotation(img, ang):
        angle = np.random.uniform(-ang, ang)
        img = ndimage.interpolation.rotate(img, angle)
        return img

    @staticmethod
    def random_full_rotation(img):
        #rotate either 0, 90, 180 or 270 degrees
        rot = np.random.randint(0, 4)
        if rot==0:
            return img
        elif rot==1:
           img, _ = utils.rotate(img, 90) 
        elif rot==2:
           img, _ = utils.rotate(img, 180) 
        elif rot==3:
           img, _ = utils.rotate(img, 270) 
        return img


    @staticmethod
    def random_crop(img, dst_shape):
        margin_0 = img.shape[0]-dst_shape[0]
        margin_1 = img.shape[1]-dst_shape[1]

        margin_0 = np.random.randint(0,margin_0)
        margin_1 = np.random.randint(0,margin_1)

        min_0 = np.random.randint(0, margin_0) if margin_0>0 else 0
        min_1 = np.random.randint(0, margin_1) if margin_1>0 else 0
        
        patch = img[min_0:(min_0+dst_shape[0]), min_1:(min_1+dst_shape[1]), :]
        return patch

 
if __name__ == '__main__':
    path = utils.get_path(data_fold=utils.TRAINING, in_or_out=utils.IN)
    roofs = DataLoader.get_all_patches_folder(folder_path=path, grayscale=False, merge_imgs=False)
    for img_name, roof_types in roofs.iteritems():
        for roof_type, roof_list in roof_types.iteritems():
            print roof_type
            if roof_type == 'metal':
                continue
            for i, roof in enumerate(roof_list):
                cv2.imwrite('debug/{}_{}_1_{}.jpg'.format(img_name, i, 'normal'), roof)
                roof = utils.resize_rgb(roof, w=utils.PATCH_H, h=utils.PATCH_W)
                #rotate it
                #roof = Augmenter().random_full_rotation(roof)
                #cv2.imwrite('debug/{}_{}_2_{}.jpg'.format(img_name, i, 'rotated'), roof)
                #flip it
                roof = Augmenter().random_flip(roof)
                cv2.imwrite('debug/{}_{}_3_{}.jpg'.format(img_name, i, 'flip'), roof)
                #crop it
                roof = Augmenter().random_crop(roof, (utils.CROP_SIZE, utils.CROP_SIZE))
                cv2.imwrite('debug/{}_{}_4_{}.jpg'.format(img_name, i, 'crop'), roof)


