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


def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

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

def save_roof_patch(img_loc, img_id, roof, roof_id):
    im = misc.imread(img_loc)
    if ((roof.ycentroid-(PATCH_H/2) < 0) or \
            (roof.ycentroid+(PATCH_H/2)-1)>im.shape[1] or \
                ((roof.xcentroid-(PATCH_W/2)) < 0) or \
                    (roof.xcentroid+(PATCH_W/2)-1)>im.shape[0]):
        pass

    else:
        patch = im[(roof.ycentroid-(PATCH_H/2)):(roof.ycentroid+(PATCH_H/2)-1),
                    (roof.xcentroid-(PATCH_W/2)):(roof.xcentroid+(PATCH_W/2)-1)]
        patch_loc = 'data/positive-patches/'+roof.roof_type+'/'+str(img_id)+'_'+str(roof_id)+'.jpg'
        misc.imsave(patch_loc, patch)


def roof_patch(img_loc, img_id, roof, roof_id, num_patch, label_file):

    print img_id
    im = misc.imread(img_loc)
    cur_type = 1 if roof.roof_type=='metal' else 2

    if ((roof.ycentroid-(PATCH_H/2) < 0) or \
            (roof.ycentroid+(PATCH_H/2)>im.shape[1]) or \
                ((roof.xcentroid-(PATCH_W/2)) < 0) or \
                    ((roof.xcentroid+(PATCH_W/2))>im.shape[0])):
        return num_patch

    w, h = roof.get_roof_size()

    #get patch the normal way, save it
    full_roof = im[(roof.ymin):(roof.ymax),(roof.xmin):(roof.xmax)]
    try:
        patch = im[(roof.ycentroid-(PATCH_H/2)):(roof.ycentroid+(PATCH_H/2)),
                    (roof.xcentroid-(PATCH_W/2)):(roof.xcentroid+(PATCH_W/2))]
        #patch_loc = 'data/testing/'+roof.roof_type+str(img_id)+'_'+str(roof_id)+'_'+str(patch_num)+'.jpg'
        patch_loc = 'data/testing/'+str(num_patch)+'.jpg'
        misc.imsave(patch_loc, patch)
        label_file.write(str(num_patch)+','+str(cur_type)+'\n')
        num_patch += 1
    except ValueError:
        pdb.set_trace()

    #if roof is too large, get multiple equally sized patches from it
    if w > PATCH_W or h > PATCH_H:
        hor_patches = w/PATCH_W
        vert_patches = h/PATCH_H
        x_pos = 0
        y_pos = 0
        patch_num += 1

        for y in range(vert_patches):
            y_pos += y*PATCH_H
            for x in range(hor_patches):
                x_pos += x*PATCH_W
                try:
                    #get all the horizontal patches
                    patch = im[(y_pos):(y_pos+PATCH_H), (x_pos):(x_pos+PATCH_W)]
                    #patch_loc = 'data/testing/'+roof.roof_type+str(img_id)+'_'+str(roof_id)+'_'+str(patch_num)+'.jpg'
                    patch_loc = 'data/testing/'+str(num_patch)+'.jpg'
                    num_patch += 1
                    misc.imsave(patch_loc, patch)
                    label_file.write(str(num_patch)+','+str(cur_type)+'\n')
                    patch_num += 1
                except ValueError:
                    pdb.set_trace()

                #plt.imshow(patch)
                #plt.show()

            if (w%PATCH_W>0) and (w>PATCH_W):
                x_pos = (hor_patches*PATCH_W)-(w%PATCH_W) #this is the leftover
                try:
                    patch = im[(y_pos):(y_pos+PATCH_H), (x_pos):(x_pos+PATCH_W)]
                    #patch_loc = 'data/testing/'+roof.roof_type+str(img_id)+'_'+str(roof_id)+'_'+str(patch_num)+'.jpg'
                    patch_loc = 'data/testing/'+str(num_patch)+'.jpg'
                    num_patch += 1
                    misc.imsave(patch_loc, patch)
                    label_file.write(str(num_patch)+','+str(cur_type)+'\n')
                    patch_num += 1
                except ValueError:
                    pdb.set_trace()
                
        '''
        # TODO: you are missing the last row!
        # if h%PATCH_H>0:
        #     y_pos = (vert_patches*PATCH_W)-(w%PATCH_W) #this is the leftover
        #     for x in range(hor_patches):
        #         patch = im[(y_pos):(y_pos+PATCH_H), (x_pos):(x_pos+PATCH_W)]
        #         patch_loc = 'data/testing/'+roof.roof_type+'/'+str(img_id)+'_'+str(roof_id)+'_'+str(patch_num)+'.jpg'
        #         misc.imsave(patch_loc, patch)
        #         patch_num += 1
            #TODO: you are missing the very last corner
        '''
    return num_patch


if __name__ == '__main__':
    img_names = set()
    inhabited_path = "data/inhabited/"
    label_loc = "data/testing/labels.txt"

    # if DEBUG: #only use one image
    #     img_names.add("0001.jpg")
    # else:
    #     #Get the filenames
    #     for file in os.listdir(inhabited_path):
    #         if file.endswith(".jpg"):
    #             img_names.add(file)

    # #Get the roofs defined in the xml, save the corresponding image patches
    # max_w = 0
    # max_h = 0
    # num_patch = 0
    # label_file = open(label_loc, 'w')

    # for i, img in enumerate(img_names):
    #     img_path = inhabited_path+img
    #     xml_path = inhabited_path+img[:-3]+'xml'

    #     roof_list, cur_w, cur_h = get_roof_positions(xml_path, max_w, max_h)
    #     # max_w = cur_w if cur_w>max_w else max_w 
    #     # max_h = cur_h if cur_h>max_h else max_h 
    #     for r, roof in enumerate(roof_list):
    #         #save_roof_patch(img_path, i+1, roof, r+1)
    #         num_patch = roof_patch(img_path, i+1, roof, r+1, num_patch, label_file)


    label_file = open(label_loc, 'a')
    num_patch = 1115
    cur_type = 0     #background patch
    img_names = set()
    uninhabited_path = "data/uninhabited/"
    if DEBUG: #only use one image
        img_names.add("0001.jpg")
    else:
        #Get the filenames
        for file in os.listdir(uninhabited_path):
            if file.endswith(".jpg"):
                img_names.add(file)

    #Get around double the number of negative patches
    for i, img in enumerate(img_names):
        for p in range((num_patch*2)/len(img_names)):
            im = misc.imread(uninhabited_path+img)
            w = im.shape[0]
            h = im.shape[1]
            w_max = w - PATCH_W
            h_max = h - PATCH_H
            xmin = random.randint(0, w_max)
            ymin = random.randint(0, h_max)
            patch = im[xmin:xmin+PATCH_W, ymin:ymin+PATCH_H]

            misc.imsave('data/testing/'+str(num_patch)+'.jpg', patch)
            label_file.write(str(num_patch)+','+str(cur_type)+'\n')
            num_patch += 1

    label_file.close()









