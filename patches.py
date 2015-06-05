from scipy import misc
import numpy as np
import matplotlib.pyplot as plt #display images
import xml.etree.ElementTree as ET #traverse the xml files
import pdb  
import os #import the image files


class Roof(object):
    def __init__(self, roof_type=None, xmin=-1, xmax=-1, ymin=-1, ymax=-1):
        self.type = roof_type
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

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

def get_roof_positions(xml_file):
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
            roof_list.append(roof)
    return roof_list

def save_roof_patch(img_loc, img_id, roof, roof_id):
    im = misc.imread(img_loc)
    #TODO: maybe the coordinates are simply upside down
    patch = im[roof.ymin:roof.ymax, roof.xmin:roof.xmax]
    patch_loc = 'data/patches/'+roof.roof_type+'/'+str(img_id)+'_'+str(roof_id)+'.jpg'
    misc.imsave(patch_loc, patch)


inhabited_path = "data/inhabited/"
#keep it simple for now
img_names = set()
for file in os.listdir(inhabited_path):
    if file.endswith(".jpg"):
        img_names.add(file)

for i, img in enumerate(img_names):
    img_path = inhabited_path+img
    xml_path = inhabited_path+img[:-3]+'xml'
    roof_list = get_roof_positions(xml_path)
    for r, roof in enumerate(roof_list):
        save_roof_patch(img_path, i+1, roof, r+1)









