from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt #display images
import xml.etree.ElementTree as ET #traverse the xml files
import pdb  
import os #import the image files
#import cv2 #for Gaussian Pyramid
import random


PATCH_W = PATCH_H = 40
DEBUG = False

class Roof(object):
    def __init__(self, roof_type=None, xmin=-1, xmax=-1, ymin=-1, ymax=-1, xcentroid=-1, ycentroid=-1):
        self.type = roof_type
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xcentroid = xcentroid
        self.ycentroid = ycentroid

    def set_centroid(self):
        self.xcentroid = self.xmin+((self.xmax - self.xmin) /2)
        self.ycentroid = self.ymin+((self.ymax - self.ymin) /2)

    def get_roof_size(self):
        return ((self.xmax-self.xmin), (self.ymax-self.ymin))

def get_roof_positions(xml_file, max_w=0, max_h=0):
    '''
    Return list of Roofs

    Parameters
    ----------
    xml_file: string
        Path to XML file that contains roof locations for current image

    Returns
    ----------
    roof_list: list
        List of roof objects in an image as stated in its xml file
    '''
    roof_list = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for child in root:
        if child.tag == 'object':
            roof = Roof()
            for grandchild in child:
                #get roof type
                if grandchild.tag == 'action':
                    roof.roof_type = grandchild.text

                #get positions of bounding box
                if grandchild.tag == 'bndbox':
                    for item in grandchild:
                        pos = int(float(item.text))
                        pos = pos if pos >= 0 else 0
                        if item.tag == 'xmax':
                            roof.xmax = pos
                        elif item.tag == 'xmin':
                            roof.xmin = pos
                        elif item.tag  == 'ymax':
                            roof.ymax = pos
                        elif item.tag  == 'ymin':
                            roof.ymin = pos
            cur_w = roof.xmax - roof.xmin
            cur_h = roof.ymax - roof.ymin
            max_w = cur_w if cur_w>max_w else max_w 
            max_h = cur_h if cur_h>max_h else max_h 
            roof.set_centroid()
            roof_list.append(roof)

    return roof_list, max_w, max_h




def roof_patch(img_loc, img_id, roof, label_file, num_patch=0):

    im = misc.imread(img_loc)
    cur_type = 1 if roof.roof_type=='metal' else 2

    if ((roof.ycentroid-(PATCH_H/2) < 0) or \
            (roof.ycentroid+(PATCH_H/2)>im.shape[1]) or \
                ((roof.xcentroid-(PATCH_W/2)) < 0) or \
                    ((roof.xcentroid+(PATCH_W/2))>im.shape[0])):
        return num_patch

    w, h = roof.get_roof_size()

    #get patch the normal way
    patch = im[(roof.ycentroid-(PATCH_H/2)):(roof.ycentroid+(PATCH_H/2)),
                (roof.xcentroid-(PATCH_W/2)):(roof.xcentroid+(PATCH_W/2))]
    misc.imsave('data/train/'+str(num_patch)+'.jpg', patch)
    label_file.write(str(num_patch)+','+str(cur_type)+'\n')
    print img_id, num_patch
    num_patch += 1

    #if roof is too large, get multiple equally sized patches from it
    if w > PATCH_W or h > PATCH_H:
        hor_patches = w/PATCH_W
        vert_patches = h/PATCH_H
        x_pos = 0
        y_pos = 0

        for y in range(vert_patches):
            y_pos += y*PATCH_H
            for x in range(hor_patches):
                x_pos += x*PATCH_W
                #get all the horizontal patches
                patch = im[(y_pos):(y_pos+PATCH_H), (x_pos):(x_pos+PATCH_W)]
                misc.imsave('data/train/'+str(num_patch)+'.jpg', patch)
                label_file.write(str(num_patch)+','+str(cur_type)+'\n')
                print img_id, num_patch
                num_patch += 1


            if (w%PATCH_W>0) and (w>PATCH_W):
                x_pos = (hor_patches*PATCH_W)-(w%PATCH_W) #this is the leftover
                patch = im[(y_pos):(y_pos+PATCH_H), (x_pos):(x_pos+PATCH_W)]
                misc.imsave('data/train/'+str(num_patch)+'.jpg', patch)
                label_file.write(str(num_patch)+','+str(cur_type)+'\n')
                print img_id, num_patch
                num_patch += 1

                
        
#TODO: you are missing the last row!
#TODO: you are missing the very last corner
    return num_patch


def get_negative_patches(num_patch, label_file):
    #get patches
    cur_type = 0     #background patch
    img_names = set()
    uninhabited_path = "data/uninhabited/"
    #Get the filenames
    for file in os.listdir(uninhabited_path):
        if file.endswith(".jpg"):
            img_names.add(file)

    negative_patches = (num_patch)/len(img_names)
    
    #Get negative patches
    for i, img in enumerate(img_names):
        for p in range(negative_patches):
            im = misc.imread(uninhabited_path+img)
            w = im.shape[0]
            h = im.shape[1]
            w_max = w - PATCH_W
            h_max = h - PATCH_H
            xmin = random.randint(0, w_max)
            ymin = random.randint(0, h_max)
            patch = im[xmin:xmin+PATCH_W, ymin:ymin+PATCH_H]

            misc.imsave('../data/train/'+str(num_patch)+'.jpg', patch)
            label_file.write(str(num_patch)+','+str(cur_type)+'\n')
            print i, num_patch
            num_patch += 1
    return num_patch


if __name__ == '__main__':
    img_names = set()
    inhabited_path = "../data/inhabited/"
    label_loc = ".../data/labels.csv"
    
    #Get the filenames
    for file in os.listdir(inhabited_path):
        if file.endswith(".jpg"):
            img_names.add(file)

    #Get the roofs defined in the xml, save the corresponding image patches
    max_w = 0
    max_h = 0
    num_patch = 0
    label_file = open(label_loc, 'w')

    for i, img in enumerate(img_names):
        img_path = inhabited_path+img
        xml_path = inhabited_path+img[:-3]+'xml'

        roof_list, cur_w, cur_h = get_roof_positions(xml_path, max_w, max_h)
        for r, roof in enumerate(roof_list):
            num_patch = roof_patch(img_path, i+1, roof, label_file, num_patch)
        if i>30:
            break
    pdb.set_trace()
    num_patch = get_negative_patches(num_patch, label_file)
    print num_patch

    label_file.close()

