import numpy as np
import pdb  
import os #import the image files
import random #get random patches
import xml.etree.ElementTree as ET #traverse the xml files
from collections import defaultdict
import pickle
import subprocess

from scipy import misc, ndimage #load images
import cv2
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.cross_validation import LeaveOneLabelOut

import json
from pprint import pprint
import utils



class DataLoader(object):
    def __init__(self, labels_path=None, out_path=None, in_path=None):
        self.total_patch_no = 0
        self.step_size = float(utils.PATCH_W)/2

        self.labels_path = labels_path
        self.out_path = out_path
        self.in_path = in_path

    @staticmethod
    def get_all_patches_folder(folder_path, grayscale=False):
        img_names = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        all_patches = dict()
        for img_name in img_names:
            all_patches[img_name] = dict()
            for roof_type in ['metal', 'thatch']:
                polygons = DataLoader.get_polygons(roof_type=roof_type, xml_path=folder_path, xml_name=img_name[:-3]+'.xml')
                all_patches[img_name][roof_type] = DataLoader.extract_patches(polygons, img_path=folder_path, grayscale=grayscale)
        return all_patches 

    @staticmethod
    def get_polygons(roof_type=None, xml_name=None, xml_path=None):
        if roof_type =='metal':
            polygon_list = DataLoader.get_metal_polygons(xml_name=xml_name)
        elif roof_type == 'thatch':
            polygon_list = DataLoader.get_thatch_polygons(xml_name=xml_name, xml_path=xml_path)
        return polygon_list


    @staticmethod
    def extract_patches(polygon_list, img_path=None, grayscale=False):
        '''
        Extract polygons from the image and array of patches
        '''
        assert img_path is not None
        try:
            img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)
        except IOError as e:
            print e
            sys.exit(-1)

        patches = list()
        for i, polygon in enumerate(polygon_list):
            print polygon
            patches.append(utils.four_point_transform(img, np.array(polygon, dtype = "float32")))
            print polygon
            pdb.set_trace()
        return patches


    @staticmethod
    def get_thatch_polygons(xml_name=None, xml_path=None):
        '''Return list of Roofs
        '''
        tree = ET.parse(xml_path+xml_name)
        root = tree.getroot()
        polygon_list = list()
        
        for child in root:
            thatch = False
            if child.tag == 'object':
                xmin, xmax, ymin, ymax = -1, -1, -1, -1
                for grandchild in child:
                    if grandchild.tag == 'action':
                        if grandchild.text[:6].lower() == 'thatch':
                            thatch = True
                        else:
                            continue 
                    #get positions of bounding box
                    if grandchild.tag == 'bndbox':
                        for item in grandchild:
                            pos = int(float(item.text))
                            pos = pos if pos >= 0 else 0
                            if item.tag == 'xmax':
                                xmax = pos
                            elif item.tag == 'xmin':
                                xmin = pos
                            elif item.tag  == 'ymax':
                                ymax = pos
                            elif item.tag  == 'ymin':
                                ymin = pos
            if thatch:
                w, h = xmax-xmin, ymax-ymin
                rect = (xmin, ymin, w, h) 
                polygon_list.append(utils.convert_rect_to_polygon(rect))
        return polygon_list


    @staticmethod
    def get_metal_polygons(xml_path=utils.RECTIFIED_COORDINATES, xml_name=None):
        '''
        Return coordinates of metal polygons in image
        '''
        assert xml_name is not None
        xml_path = xml_path+xml_name

        #EXTRACT THE POLYGONS FROM THE XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        polygon_list = list()
        
        for child in root:
            if child.tag == 'object':
                for grandchild in child:
                    #get positions of bounding box
                    if grandchild.tag == 'polygon':
                        polygon = list() #list of four points

                        for coordinates in grandchild:
                            if coordinates.tag == 'pt':
                                for point in coordinates:
                                    pos = int(float(point.text))
                                    pos = pos if pos >= 0 else 0
                                    if point.tag == 'x':
                                        x = pos
                                    elif point.tag == 'y':
                                        y = pos
                                polygon.append((x,y))
                        if len(polygon) == 4:
                            polygon_list.append(polygon)
        return np.array(polygon_list)


#######################################################################
## SEPARATING THE DATA INTO TRAIN, VALIDATION AND TESTING SETS
#######################################################################
    def get_train_test_valid_all(self, original_data_only=False):
        '''Will write to either test, train of validation folder each of the images from source
        '''
        groups = list()
                
        #get list of images in source/inhabted and source/inhabited_2
        train_imgs = [utils.INHABITED_1+img for img in DataLoader.get_img_names_from_path(path =utils.INHABITED_1)]
        if original_data_only == False:
            train_2 = [utils.INHABITED_2+img for img in DataLoader.get_img_names_from_path(path =utils.INHABITED_2)]
            train_imgs = train_imgs + train_2

        #shuffle for randomness
        train_imgs = shuffle(train_imgs, random_state=0)

        total_metal = total_thatch = 0
        roof_dict = dict()
        for i, img_path in enumerate(train_imgs): 
            roofs = self.get_roofs(img_path[:-3]+'xml', img_path)
            cur_metal= sum([0 if roof.roof_type=='thatch' else 1 for roof in roofs ])
            cur_thatch= sum([0 if roof.roof_type=='metal' else 1 for roof in roofs ])
            total_metal += cur_metal
            total_thatch += cur_thatch
            roof_dict[img_path] = (cur_metal, cur_thatch) 

        train_imgs, train_imgs_left, metal_left_over, thatch_left_over, train_metal, train_thatch = self.get_50_percent(roof_dict, 
                                                                                                            train_imgs, total_metal, total_thatch)
        valid_imgs, test_imgs, test_metal, test_thatch, valid_metal, valid_thatch = self.get_50_percent(roof_dict, train_imgs_left, metal_left_over, thatch_left_over)

        train_path = utils.ORIGINAL_TRAINING_PATH if original_data_only else utils.TRAINING_PATH
        valid_path = utils.ORIGINAL_VALIDATION_PATH if original_data_only else utils.VALIDATION_PATH
        testing_path = utils.ORIGINAL_TESTING_PATH if original_data_only else utils.TESTING_PATH
        for files, dest in zip([train_imgs, valid_imgs, test_imgs], [train_path, valid_path, testing_path]):
            for img in files:
                print 'Saving file {0} to {1}'.format(img, dest)
                subprocess.check_call('cp {0} {1}'.format(img, dest), shell=True)
                subprocess.check_call('cp {0} {1}'.format(img[:-3]+'xml', dest), shell=True)

        out_path = '../data_original/' if original_data_only else '../data/'
        with open(out_path+'data_stats.txt' , 'w') as r:
            r.write('\tTrain\tValid\tTest\n')
            r.write('Metal\t{0}\t{1}\t{2}\n'.format(train_metal, valid_metal, test_metal))
            r.write('Thatch\t{0}\t{1}\t{2}\n'.format(train_thatch, valid_thatch, test_thatch))
                

    def get_50_percent(self, roof_dict, train_imgs, total_metal, total_thatch):
        # we want to keep around 50:25:25 ratio for training, validation, testing
        metal_40 = int(0.48*total_metal)        
        thatch_40 = int(0.48*total_thatch)
        metal_60 = int(0.55*total_metal)
        thatch_60 = int(0.55*total_thatch)

        cumulative_metal = 0
        cumulative_thatch = 0
        img_index = -1
        for i, img_path in enumerate(train_imgs):
            cumulative_metal += roof_dict[img_path][0]
            cumulative_thatch += roof_dict[img_path][1]
            if (cumulative_metal>metal_40 and cumulative_thatch>metal_40) and (cumulative_metal<metal_60 and cumulative_thatch<thatch_60):
                img_index = i
                break
        assert img_index != -1
        return train_imgs[:i+1], train_imgs[i+1:], total_metal-cumulative_metal, total_thatch-cumulative_thatch, cumulative_metal, cumulative_thatch 



if __name__ == '__main__':
    pass    
            




