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
import cv
from scipy import misc, ndimage #load images

import get_data
from get_data import DataAugmentation
import experiment_settings as settings


class ViolaDataSetup(object):
#######################################################
##### FINAL METHOD USED TO PRODUCE PATCHES FOR VIOLA
#######################################################
    @staticmethod
    def setup_negative_samples():
        '''Write file with info about location of negative examples
        '''
        output = open(settings.BG_FILE, 'w')
        for file in os.listdir(settings.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(settings.UNINHABITED_PATH+file+'\n')


    @staticmethod
    def original_dataset_setup_single_size_patches_augment(in_path=settings.ORIGINAL_TRAINING_PATH, out_path=''):
        '''
        NOT RECTIFIED
        No division between different roof sizes: if a roof has a size that is off, we resize it
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''
        assert out_path != ''
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=in_path)
        roof_loader = get_data.DataLoader()
        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            xml_path = in_path+img_name[:-3]+'xml'
            img_path = in_path+img_name
    
            roofs = roof_loader.get_roofs(xml_path, img_name)
            metal_log = list()
            thatch_log = list()
            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
            except IOError:
                print 'Cannot open '+img_path
            else:
                for roof_id, roof in enumerate(roofs):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)
                    roof_type = roof.roof_type 
                    roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])
                       
                    #if it's vertical, make it lie down
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    
 
                    #write basic positive example to the right folder
                    img_name_short = img_name[:-4]
                    general_path = '{0}{1}/{2}_{3}'.format(out_path, roof_type, img_name_short, roof_id)
                    try:
                        cv2.imwrite('{0}.jpg'.format(general_path), roof_img)    
                    except Exception as e:
                        print e
                        pdb.set_trace()
                    #calculate and write the augmented images too
                    DataAugmentation.flip_pad_save(img_path, roof, general_path)


    @staticmethod
    def rectified_metal_original_dataset_augment_single_size_patches(in_path=settings.ORIGINAL_TRAINING_PATH, out_path=settings.ORIGINAL_VIOLA_RECT_AUGM):
        '''
        RECTIFIED, METAL ONLY, ORIGINAL DATASET
        No division between different roof sizes: if a roof has a size that is off, we resize it
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''

        img_names_list = get_data.DataLoader().get_img_names_from_path(path=in_path)
        roof_loader = get_data.DataLoader()
        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            img_path = in_path+img_name
            roof_patches = get_data.DataLoader().get_roof_patches_from_rectified_dataset(xml_name=img_name[:-3]+'xml', img_path=img_path)

            for roof_id, roof_img in enumerate(roof_patches):
                print 'Processing image {0}: roof {1}'.format(img_id, roof_id)
                   
                #if it's vertical, make it lie down
                if roof_img.shape[0] > roof_img.shape[1]:
                    roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                

                #write basic positive example to the right folder
                img_name_short = img_name[:-4]+'_metal'
                general_path = '{0}{1}_{2}'.format(out_path, img_name_short, roof_id)
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


    @staticmethod
    def dat_file_setup_positive_samples_full_image_single_size(in_path=None, dat_file_name=None):
        '''Save .dat files containing location of training images
        Each of the training images contains a single roof
        '''
        assert in_path is not None
        assert dat_file_name is not None
        try:
            dat_f = open(settings.DAT_PATH+dat_file_name, 'w')
        except IOError as e:
            print e
            sys.exit(-1)
        else:
            img_names_list = get_data.DataLoader().get_img_names_from_path(path=in_path)
            for img_name in img_names_list:
                img_path = in_path+img_name
                try:
                    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                except IOError:
                    print 'Cannot open '+img_path
                else:
                    height, width, _ = img.shape

                    #append roof characteristics separated by single space
                    position_string = '{0} {1} {2} {3}'.format(0, 0, width, height)
                    log_to_file = '{0}\t{1}\t{2}\n'.format('../'+img_path, 1, position_string)

                    #add to .dat file depending on roof time and on ratio (if metal)
                    dat_f.write(log_to_file)
        finally:
            dat_f.close()


    @staticmethod
    def get_roof_dat(roof_lists, roof, size_divide=False):
        '''Return string with roof position and size for .dat file

        Roof is added to list depending on width/height ratio
        '''
        string_to_add = '{0} {1} {2} {3}'.format(roof.xmin, roof.ymin, roof.width, roof.height)
        #if we care about roof size, add the roofs to the right list depending on their height/width ratio
        ratio = 0
        if roof.roof_type == 'metal' and size_divide:
            aspect_ratio = float(roof.width)/(roof.height)
            if (aspect_ratio > 1.25):                       #WIDE ROOF
                roof_lists[2].append(string_to_add)
            elif aspect_ratio >= 0.75 and ratio <= 1.25:    #SQUARE
                roof_lists[1].append(string_to_add)
            elif aspect_ratio < 0.75:                      #TALL ROOF
                roof_lists[0].append(string_to_add)
        else:#either it's a thatches roof or it's metal but we don't care about the width/height ratio
            roof_lists.append(string_to_add)


    @staticmethod
    def get_dat_string(roof_list, img_path):
        '''Return string formatted for .dat file
        '''
        return img_path+'\t'+str(len(roof_list))+'\t'+'\t'.join(roof_list)+'\n'
    

#################################################################
#VEC FILE SETUP
################################################################
    @staticmethod
    def vec_file_samples():
        '''Save a vec file for every .dat file found in settings.DAT_PATH
        '''
        vec_info = list()

        for file_name in os.listdir(settings.DAT_PATH):
            if file_name.endswith('.dat'):
                # number of lines tells us number of samples
                sample_num = ViolaDataSetup.get_sample_num_from_dat(settings.DAT_PATH+file_name) 
                
                w = raw_input('Width of {0}: '.format(file_name))
                h = raw_input('Height of {0}: '.format(file_name))
                
                vec_file_name = '{0}_num{1}_w{2}_h{3}.vec'.format(file_name[:-4], sample_num, w, h)
                vec_file = settings.VEC_PATH+vec_file_name
                vec_info.append( (vec_file_name, int(float(sample_num)), int(float(w)), int(float(h))) )

                dat_file = settings.DAT_PATH+file_name
                sample_cmd ='/usr/bin/opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_file, settings.BG_FILE, vec_file, sample_num, w, h)
                try:
                    subprocess.check_call(sample_cmd, shell=True)
                    #move_cmd = 'mv {0} ../viola_jones/all_dat/'.format(dat_file)
                    #subprocess.check_call(move_cmd, shell=True)
                except Exception as e:
                    print e
        return vec_info


    @staticmethod
    def get_sample_num_from_dat(dat_path):
        with open(dat_path, 'rb') as csvfile:
            sample_num = 0
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                sample_num += int(row[1])
        return sample_num


    @staticmethod
    def vec_file_single(dat_file=''):
        '''Produce vec file for given dat file
        '''

        dat_path = '{0}{1}'.format(settings.DAT_PATH, dat_file)
        sample_num = ViolaDataSetup.get_sample_num_from_dat(dat_path)
        w = raw_input('Width of {0}: '.format(dat_file[:-4]))
        h = raw_input('Height of {0}: '.format(dat_file[:-4]))

        vec_file = '{0}_num{3}_w{1}_h{2}.vec'.format(dat_file[:-4], w, h, sample_num)
        vec_path = '{0}{1}'.format(settings.VEC_PATH, vec_file)

        vec_cmd ='/usr/bin/opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_path, settings.BG_FILE, vec_path, sample_num, w, h)
        try:
            subprocess.check_call(vec_cmd, shell=True)
        except Exception as e:
            print e 
        return [(vec_file, sample_num, w, h)]


########################################
## FINAL CODE USED BELOW
########################################

def data_setup():
    ViolaDataSetup.setup_negative_samples() 
    dat_names = ViolaDataSetup.setup_positive_samples(padding=0, path=settings.TRAINING_PATH, size_divide=False)     
    ViolaDataSetup.vec_file_samples()

    
def original_dataset_setup_patches_and_dat_file(dat_file=None):
    '''
    This is the simpler version that uses all roofs in one batch, only divided by roof type
    '''
    assert dat_file is not None
    ViolaDataSetup.original_dataset_setup_single_size_patches_augment()
    in_path = ORIGINAL_VIOLA_PATCHES_AUGM
    for roof_type in ['metal', 'thatch']:
        dat_file = '{0}_original_singlesize_augm1.dat'.format(roof_type)
        ViolaDataSetup.dat_file_setup_positive_samples_full_image_single_size(in_path=in_path, dat_file_name= dat_file)


if __name__ == '__main__':
    #ViolaDataSetup.rectified_metal_original_dataset_augment_single_size_patches()
    ViolaDataSetup.dat_file_setup_positive_samples_full_image_single_size(in_path=settings.ORIGINAL_VIOLA_RECT_AUGM, dat_file_name='metal_rect_augm1_singlesize_original_pad0.dat')  
    ViolaDataSetup.vec_file_single(dat_file='metal_rect_augm1_singlesize_original_pad0.dat')

