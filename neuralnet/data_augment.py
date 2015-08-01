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
    def random_rotation(self, ang, fill_mode="nearest", cval=0.):
        angle = np.random.uniform(-ang, ang)
        self.Xb = scipy.ndimage.interpolation.rotate(self.Xb, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)


    @staticmethod
    def random_crop(img, dst_shape):
        margin_0 = img.shape[0]-dst_shape[0]
        margin_1 = img.shape[1]-dst_shape[1]
        min_0 = np.random.randint(0, margin_0)
        min_1 = np.random.randint(0, margin_1)
        patch = img[min_0:(min_0+dest_shape[0]), min_1:(min_1+dst_shape[1]), :]
        return patch

    '''
    @staticmethod
    def random_illumination(img, illum_range):
        img_hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)         
        percent = np.random.uniform(.50, 1.5)       
        img_hsv[:, :, 2] = img_hsv[:, :, 3]*percent

        img = cv2.cvtColor(img, cv2.cv.CV_HSV2BGR)
        return img
    '''
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

 
