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
from extract_rect import four_point_transform
import utils



class Roof(object):
    '''Roof class containing info about its location in an image
    '''
    def __init__(self, roof_type=None, xmin=-1, xmax=-1, ymin=-1, ymax=-1, xcentroid=-1, ycentroid=-1):
        self.roof_type = roof_type
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        # self.xcentroid = xcentroid
        # self.ycentroid = ycentroid


    def __comp__(self, other):
        return self.__dict__ == other.__dict__


    def set_centroid(self):
        self.xcentroid = self.xmin+((self.xmax - self.xmin) /2)
        self.ycentroid = self.ymin+((self.ymax - self.ymin) /2)
    
    def get_roof_size(self):
        return ((self.ymax-self.ymin), (self.xmax-self.xmin))

    @property 
    def height(self):
        return (self.ymax-self.ymin)

    @property 
    def width(self):
        return (self.xmax-self.xmin)


               


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


class DataLoader(object):
    def __init__(self, labels_path=None, out_path=None, in_path=None):
        self.total_patch_no = 0
        self.step_size = float(utils.PATCH_W)/2

        self.labels_path = labels_path
        self.out_path = out_path
        self.in_path = in_path


    def get_roofs_json(self, json_file):
        '''
        Return a list of roofs from a json file. We only have the centroids for these, not xmax, ymax
        '''
        img_dict = defaultdict()
        with open(json_file) as data_file:
            for line in data_file:
                img_json = json.loads(line)
                img_dict[img_json['image'].strip()] = img_json

        roof_dict = defaultdict(list)
        for img_name, img_info in img_dict.items():
            roof_list = img_info['roofs']

            for r in roof_list:
                roof_type = 'metal' if str(r['type']).strip() == 'iron' else 'thatch'
                roof = Roof(roof_type=roof_type, 
                            xmin=int(float(r['x'])), xmax=int(float(r['x'])),
                            ymin=int(float(r['y'])), ymax=int(float(r['y'])))

                padding = float(utils.PATCH_W)/2
                roof = DataAugmentation.add_padding(roof, padding, utils.JSON_IMGS+img_name)
                roof_dict[str(img_name)].append(roof)

        return roof_dict


    def mark_detections_on_img(self, img_name=None, roofs=None):
        ''' Save an image with the detections and the ground truth roofs marked with rectangles
        '''
        assert img_name is not None and roofs is not None
        try:
            img = cv2.imread(self.in_path+img_name)
        except Exception, e:
            print e
            sys.exit(-1)
        else:
            for i, roof in enumerate(roofs):
                if roof.roof_type=='metal':
                    color=(255,0,0)
                elif roof.roof_type=='thatch':
                    color=(0,0,255)
                cv2.rectangle(img,(roof.xmin,roof.ymin),(roof.xmin+roof.width,roof.ymin+roof.height), color, 2)
            cv2.imwrite(self.out_path+'0_FULL_'+img_name[:-4]+'.jpg', img)

        return img


    def get_roofs(self, xml_file, img_name):
        '''Return list of Roofs

        Parameters
        ----------
        xml_file: string
            Path to XML file that contains roof locations for current image
        img_name: string
            Name of the current image, so roof knows which image it comes from

        Returns
        ----------
        roof_list: list
            List of roof objects in an image as stated in its xml file
        max_w, max_h: int
            The integer value of the maximum width and height of all roofs in the current image
        '''
        tree = ET.parse(xml_file)
        root = tree.getroot()
        roof_list = list()
        
        for child in root:
            if child.tag == 'object':
                roof = Roof()
                real_roof = False
                roof.img_name = img_name 
                for grandchild in child:
                    #get roof type
                    if grandchild.tag == 'action':
                        if grandchild.text[:5].lower() == 'metal':
                            roof.roof_type = 'metal'
                            real_roof = True
                        elif grandchild.text[:6].lower() == 'thatch':
                            roof.roof_type = 'thatch'
                            real_roof = True
                        else:
                            real_roof = False
                            continue 
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
                if real_roof:
                    roof.set_centroid()
                    roof_list.append(roof)
        return roof_list



    def get_roof_patches_from_rectified_dataset(self, coordinates_only=False, xml_path=utils.RECTIFIED_COORDINATES, xml_name=None, img_path=None):
        '''
        Return roof patches from the dataset that has roofs properly bounded
        If coordinated_only is True, return only the coordinates instead of the patches
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

        if coordinates_only:
            return polygon_list
        else:
            #EXTRACT THE RECTIFIED ROOF PATCH FROM THE IMG, RETURN THE ROOF PATCHES
            assert img_path is not None
            try:
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_equalized = cv2.equalizeHist(gray)
            except IOError as e:
                print e
                sys.exit(-1)
            roof_patches = list()
            for i, polygon in enumerate(polygon_list):
                roof_patches.append(four_point_transform(gray_equalized, np.array(polygon, dtype = "float32")))
            return roof_patches 



    def get_roof_imgs(self, roof_list, img_path, padding):
        try:
            img = cv2.imread(img_path)
        except Exception, e:
            print e
        else:
            for roof in roof_list:
                roof_patch = img[roof.ymin-pad:roof.ymin+roof.height+pad, roof.xmin-pad:roof.xmin+roof.width+pad]         
                

    @staticmethod
    def get_img_names_from_path(path='', extension='.jpg'):
        ''' Get list of image files in a given folder path
        '''
        img_names = set()
        for file in os.listdir(path):
            if file.endswith(extension):
                img_names.add(file)
        return img_names


    def save_patch(self, img=None, xmin=-1, ymin=-1, roof_type=-1, extension='.jpg'):
        '''Save a patch located at the xmin, ymin position to the patches folder 
        '''
        assert xmin > -1 and ymin > -1
        with open(self.labels_path, 'a') as labels_file:
            try:
                patch = img[ymin:(ymin+utils.PATCH_H), xmin:(xmin+utils.PATCH_W)]
                cv2.imwrite(self.out_path+str(self.total_patch_no)+str(extension), patch)
            except (IndexError, IOError, KeyError, ValueError) as e:
                print e
            else:
                labels_file.write(str(self.total_patch_no)+','+str(roof_type)+'\n')
                #print 'Saved patch: '+str(self.total_patch_no)+'\n'
                self.total_patch_no = self.total_patch_no+1


    def save_patch_scaled(self, roof=None, img=None, extension='.jpg'):
        '''Save a downscaled image of a roof that is too large as is to fit within the PATCH_SIZE 
        '''
        roof_type = utils.METAL if roof.roof_type=='metal' else utils.THATCH

        h, w = roof.get_roof_size()
        #TODO: this might not be accurate: it doesn't make sense to scale up/down depending on...
        max_direction = max(w, h)
        xmin = roof.xcentroid-(max_direction/2)
        ymin = roof.ycentroid-(max_direction/2)
        xmax = roof.xcentroid+(max_direction/2)
        ymax = roof.ycentroid+(max_direction/2)

        with open(self.labels_path, 'a') as labels_file:
            try:
                patch = img[ymin:ymax, xmin:xmax]
                patch_scaled = misc.imresize(patch, (utils.PATCH_H, utils.PATCH_W))
                misc.imsave(self.out_path+str(self.total_patch_no)+extension, patch_scaled)

            except (IndexError, IOError, KeyError, ValueError) as e:
                print e

            else:
                labels_file.write(str(self.total_patch_no)+','+str(roof_type)+'\n')
                self.total_patch_no = self.total_patch_no+1


    def get_horizontal_patches(self, img=None, roof=None, y_pos=-1):
        '''Get patches along the width of a roof for a given y_pos (i.e. a given height in the image)
        '''
        roof_type = utils.METAL if roof.roof_type=='metal' else utils.THATCH

        h, w = roof.get_roof_size()
        hor_patches = ((w - utils.PATCH_W) // self.step_size) + 1  #hor_patches = w/utils.PATCH_W


        for horizontal in range(int(hor_patches)):
            x_pos = roof.xmin+(horizontal*self.step_size)
            self.save_patch(img= img, xmin=x_pos, ymin=y_pos, roof_type=roof_type)


    def produce_roof_patches(self, img_path=None, img_id=-1, roof=None):
        '''Given a roof in an image, produce a patch or set of patches for it
        
        Parameters:
        ----------------
        img_loc: string path
            Path of image from which to obtain patches
        img_id: int
            id of the current image
        roof: Roof type
        label_file: 
               
        Returns:
        ---------------
        total_patch_no: int
            The number of patches produced so far
        '''
        assert img_id > -1 and roof is not None and img_path is not None

        try:
            img = cv2.imread(img_path)

        except IOError:
            print 'Cannot open '+img_path

        else:
            roof_type = utils.METAL if roof.roof_type=='metal' else utils.THATCH

            ybottom = (roof.ycentroid-(utils.PATCH_H/2) < 0)
            ytop = (roof.ycentroid+(utils.PATCH_H/2)>img.shape[0])
            xbottom = ((roof.xcentroid-(utils.PATCH_W/2)) < 0) 
            xtop = ((roof.xcentroid+(utils.PATCH_W/2))>img.shape[1])

            if ybottom or xbottom or ytop or xtop:
                #could not produce any patches
                print 'Failed to produce a patch \n'
                return -1

            h, w = roof.get_roof_size()

            #get a patch at the center
            self.save_patch(img= img, xmin=(roof.xcentroid-(utils.PATCH_W/2)), 
                            ymin=(roof.ycentroid-(utils.PATCH_H/2)), roof_type=roof_type)

            # if roof is too large, get multiple equally sized patches from it, and a scaled down patch also
            if w > utils.PATCH_W or h > utils.PATCH_H:
                x_pos = roof.xmin
                y_pos = roof.ymin

                vert_patches = ((h - utils.PATCH_H) // self.step_size) + 1  #vert_patches = h/utils.PATCH_H

                #get patches along roof height
                for vertical in range(int(vert_patches)):
                    y_pos = roof.ymin+(vertical*self.step_size)
                    #along roof's width
                    self.get_horizontal_patches(img=img, roof=roof, y_pos=y_pos)

                #get patches from the last row also
                if (h % utils.PATCH_W>0) and (h > utils.PATCH_H):
                    leftover = h-(vert_patches*utils.PATCH_H)
                    y_pos = y_pos-leftover
                    self.get_horizontal_patches(img=img, roof=roof, y_pos=y_pos)

                #add an image of entire roof, scaled down so it fits within patch size
                self.save_patch_scaled(img=img, roof=roof)
                

    def get_negative_patches(self, total_patches, label_file, out_path=None):
        '''Save negative patches to a folder
        '''
        print 'Getting the negative patches....\n'
        img_names = DataLoader.get_img_names_from_path(path=utils.UNINHABITED_PATH)
        
        self.labels_path = label_file
        self.out_path = out_path if out_path is not None else self.out_path
        print 'Saving negative: '+str(total_patches)+'\n'
        negative_patches = (total_patches)/len(img_names)
        
        #Get negative patches
        for i, img_path in enumerate(img_names):
            print ('Negative image: '+str(i)+'\n')
            for p in range(negative_patches):
                #get random ymin, xmin, but ensure the patch will fall inside of the image
                try:
                    img = cv2.imread(utils.UNINHABITED_PATH+img_path)
                except IOError:
                    print 'Cannot open '+img_path
                else:
                    h, w, _ = img.shape
                    w_max = w - utils.PATCH_W
                    h_max = h - utils.PATCH_H
                    xmin = random.randint(0, w_max)
                    ymin = random.randint(0, h_max)

                    #save patch
                    self.save_patch(img=img, xmin=xmin, ymin=ymin, roof_type=utils.NON_ROOF) 
        
        

#######################################################################
## Get roofs from training set and save positive patches to the neural training folder
#######################################################################
    def neural_training_positive_full_roofs(self, pad=0, out_path=None, in_path=None):
        '''Uses images in utils/INHABITED_PATH or ../data/testing_new_source and the corresponding xml file to get roofs with padding and saves them to an 
        output folder. 
        If negatives is True, is also includes negatives.
        Aside from the images, it saves: a file with labels to indicate each roof type and a pickled dump of all the roof info

        Parameters:
        ----
        pad: int
            Pixel padding for roof patch bounding box
        negatives: boolean
            Decides whether negatives will be processed or not
        in_path: 
            Where the positive example folder is
        out_path:
            Where roofs should be saved to
        '''
        raise ValueError('what should in and out paths be? Before it was in_path=TRAINING_PATH and out_path=utils.TRAINING_NEURAL_POS')
        img_names = DataLoader.get_img_names_from_path(path=in_path)
        all_labels = list()
        all_roofs = list()
        for i, img_name in enumerate(img_names):
            img_path = in_path+img_name
            xml_path = in_path+img_name[:-3]+'xml'

            roof_objects = loader.get_roofs(xml_path, img_name)
            try:
                img = cv2.imread(img_path)
            except Exception, e:
                print e
            else: 
                all_roofs, all_labels = self.get_padded_roofs(img=img, pad=pad, roof_list=roof_objects,  
                                                                all_roofs=all_roofs, all_labels=all_labels)   
        
        all_roofs = np.array(all_roofs)
        all_labels = np.array(all_labels)
        
        #save all the patches, pickle the roofs
        with open(out_path+'labels.csv', 'w') as labels_train:
            for r, (roof, label) in enumerate(zip(all_roofs, all_labels)):
                print '{0}/{1}'.format(r, len(all_roofs))
                roof_name = '{0}.jpg'.format(r)
                out_img = out_path+roof_name
                cv2.imwrite(out_img, roof)
                labels_train.write('{0}, {1}\n'.format(r, label))
        with open(out_path+'stats.txt', 'w') as stats:
            non_roof, metal, thatch = np.bincount(all_labels)
            stats.write('non_roof,{0}\nmetal,{1}\nthatch,{2}'.format(non_roof, metal, thatch))


    def get_padded_roofs(self, img=None, pad=0, roof_list=None, all_roofs=None, all_labels=None):
        '''Get roofs using bounding boxes. Add padding. Resize the patch to match the size of a utils.PATCH
        '''
        for roof in roof_list:
            roof_type = utils.METAL if roof.roof_type=='metal' else utils.THATCH

            #get the right padding
            pad_h = utils.PATCH_H if roof.height< utils.PATCH_H else roof.height 
            pad_w = utils.PATCH_W if roof.width < utils.PATCH_W else roof.width
            pad_left = pad_right = pad_top = pad_bottom = pad
           
            #Adapt the padding to make sure we don't go off image
            offset = roof.ymin-pad_bottom
            pad_bottom = (pad_bottom + offset) if offset < 0 else pad_bottom
            
            offset = roof.xmin - pad_left
            pad_left = (pad_left + offset) if offset < 0 else pad_left

            offset = img.shape[0]-(roof.ymin + pad_h + pad_top)
            pad_top = (pad_top + offset) if offset < 0 else pad_top

            offset = img.shape[1]- (roof.xmin + pad_w + pad_right)
            pad_right = (pad_right + offset) if offset < 0 else pad_right
            
            #get the patch, resize it and append it to list
            patch = img[roof.ymin-pad_bottom:roof.ymin+pad_h+pad_top,
                        roof.xmin-pad_left:roof.xmin+pad_w+pad_right]
            resized_patch = cv2.resize(patch, (utils.PATCH_H, utils.PATCH_W), interpolation=cv2.INTER_AREA) 
            all_roofs.append(resized_patch)
            all_labels.append(roof_type)

        return all_roofs, all_labels


    def produce_json_roofs(self, json_file=None):
        #Get the filename 
        img_names = DataLoader.get_img_names_from_path(path=utils.JSON_IMGS, extension='.png')
        roof_dict = self.get_roofs_json(json_file)
        
        with open(self.labels_path, 'w') as label_file:
            for i, img in enumerate(img_names, 1):
                print 'Procesing image {0}, number {1}'.format(img, i)
                self.mark_detections_on_img(img_name=img, roofs=roof_dict[img])
                
                # img_path = self.in_path+img
                # roof_list = roof_dict[img]

                # for r, roof in enumerate(roof_list):
                #     self.produce_roof_patches(img_path=img_path, img_id=i, roof=roof)


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
    #loader = DataLoader(labels_path='labels.csv', out_path='../data/testing_json/', in_path=utils.JSON_IMGS)
    #loader.produce_json_roofs(json_file='../data/images-new/labels.json')
    #loader.get_train_test_valid_all( original_data_only=True)
     
    #loader = DataLoader()
    #loader.get_negative_patches(10000, '../data/neural_training/negatives/labels.csv', out_path='../data/neural_training/negatives/') 
    #loader.neural_training_positive_full_roofs()

    #set up the original training set, using inhabited imgs only
    DataLoader().get_train_test_valid_all(original_data_only = True) 








