import numpy as np
import pdb  
import os #import the image files
import random #get random patches
import xml.etree.ElementTree as ET #traverse the xml files
from collections import defaultdict
import pickle
import itertools
import subprocess

from scipy import misc, ndimage #load images
import cv2
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.cross_validation import LeaveOneLabelOut

import json
from pprint import pprint
import utils


class DataAugmentation(object):
###################
#Neural Augmentation
###################
    def transform(self, Xb):
        self.Xb = Xb
        self.random_flip()
        self.random_rotation(10.)
        self.random_crop()
        return self.Xb

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


####################
# Viola Jones augmentation
#####################
    @staticmethod
    def flip_pad_save(in_path, roof, img_path, equalize=True):
        for i in range(4):
            try:        
                img = cv2.imread(in_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if equalize:
                    gray = cv2.equalizeHist(gray)
            except IOError:
                print 'Cannot open '+img_path
            else:
                pad_rand = random.random()
                if pad_rand < 0.2:
                    padding = 2 
                elif pad_rand < 0.4:
                    padding = 3 
                elif pad_rand < 0.6:
                    padding = 4
                elif pad_rand < 0.8:
                    padding = 6
                else:
                    padding = 8 
                roof = DataAugmentation.add_padding(roof, padding, in_path)
                roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])

                if i == 0:
                    cv2.imwrite('{0}_flip0.jpg'.format(img_path), roof_img)
                if i == 1:
                    roof_img = cv2.flip(roof_img,flipCode=0)
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    cv2.imwrite('{0}_flip1.jpg'.format(img_path), roof_img)
                elif i == 2:
                    roof_img = cv2.flip(roof_img,flipCode=1)
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    cv2.imwrite('{0}_flip2.jpg'.format(img_path), roof_img)
                else:
                    roof_img = cv2.flip(roof_img,flipCode=-1)
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    cv2.imwrite('{0}_flip3.jpg'.format(img_path), roof_img)
            
    @staticmethod
    def flip_pad_save_metal_rect(img_path, equalize=True):
        for i in range(4):
            try:        
                img = cv2.imread(in_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if equalize:
                    gray = cv2.equalizeHist(gray)
            except IOError:
                print 'Cannot open '+img_path
            else:
                pad_rand = random.random()
                if pad_rand < 0.2:
                    padding = 2 
                elif pad_rand < 0.4:
                    padding = 3 
                elif pad_rand < 0.6:
                    padding = 4
                elif pad_rand < 0.8:
                    padding = 6
                else:
                    padding = 8 
                roof = DataAugmentation.add_padding(roof, padding, in_path)
                roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])

                if i == 0:
                    cv2.imwrite('{0}_flip0.jpg'.format(img_path), roof_img)
                if i == 1:
                    roof_img = cv2.flip(roof_img,flipCode=0)
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    cv2.imwrite('{0}_flip1.jpg'.format(img_path), roof_img)
                elif i == 2:
                    roof_img = cv2.flip(roof_img,flipCode=1)
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    cv2.imwrite('{0}_flip2.jpg'.format(img_path), roof_img)
                else:
                    roof_img = cv2.flip(roof_img,flipCode=-1)
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    cv2.imwrite('{0}_flip3.jpg'.format(img_path), roof_img)
 

    @staticmethod
    def rotateImage(img, clockwise=True):
        #timg = np.zeros(img.shape[1],img.shape[0]) # transposed image
        if clockwise:
            # rotate counter-clockwise
            timg = cv2.transpose(img)
            cv2.flip(timg,flipCode=0)
            return timg
        else:
            # rotate clockwise
            timg = cv2.transpose(img)
            cv2.flip(timg,flipCode=1)
            return timg


    @staticmethod
    def add_padding(roof, padding, img_path):        
        try:
            img = cv2.imread(img_path)
        except IOError:
            print 'Cannot open '+img_path
        else:
            img_height, img_width, _ = img.shape
            if roof.xmin-padding > 0:
                roof.xmin -= padding
            if roof.ymin-padding > 0:
                roof.ymin -= padding
            if roof.xmin+roof.width+padding < img_width:
                roof.xmax += padding
            if roof.ymin+roof.height+padding < img_height:
                roof.ymax += padding
        roof.set_centroid()
        return roof

        
