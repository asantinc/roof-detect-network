import numpy as np
import pdb  
import os #import the image files
import random #get random patches
import pickle
import itertools
import subprocess

import cv2
import utils


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
    def random_rotation(x, ang):
        angle = np.random.uniform(-ang, ang)
        x = ndimage.interpolation.rotate(x, angle)
        return x

    @staticmethod
    def random_full_rotation(x):
        #rotate either 0, 90, 180 or 270 degrees
        rot = np.random.randint(0, 4)
        if rot==0:
            return x
        elif rot==1:
           x = utils.rotate(x, 90) 
        elif rot==2:
           x = utils.rotate(x, 180) 
        elif rot==3:
           x = utils.rotate(x, 270) 


    @staticmethod
    def random_crop(img, dst_shape):
        margin_0 = img.shape[0]-dst_shape[0]
        margin_1 = img.shape[1]-dst_shape[1]
        min_0 = np.random.randint(0, margin_0)
        min_1 = np.random.randint(0, margin_1)
        patch = img[min_0:(min_0+dest_shape[0]), min_1:(min_1+dst_shape[1]), :]
        return patch

 
if __name__ == '__main__':
    path = utils.get_path(data_fold=utils.TRAINING, in_or_out=utils.IN)
    roofs = get_all_patches_folder(folder_path=path, grayscale=False, merge_imgs=False)
    for img_name, roof_types in roofs.iteritems():
        for roof_type, roof in roof_types.iteritems():
            cv2.imsave('{}_{}.jpg'.format(img_name, 'normal')
            roof = utils.resize(roof, utils.PATCH_H, utils.PATCH_W)
            #rotate it
            roof = Augmenter().random_full_rotation(roof)
            cv2.imsave('{}_{}.jpg'.format(img_name, 'rotated')
            #flip it
            roof = Augmenter().random_flip(roof)
            cv2.imsave('{}_{}.jpg'.format(img_name, 'flip')
            #crop it
            roof = Augmented().random_crop(roof, (utils.CROP_SIZE, utils.CROP_SIZE))
            cv2.imsave('{}_{}.jpg'.format(img_name, 'crop')
            pdb.set_trace()


