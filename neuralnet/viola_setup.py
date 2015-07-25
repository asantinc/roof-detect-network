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
from scipy import misc, ndimage #load images

import get_data
from get_data import DataLoader
from augmentation import DataAugmentation
import utils


class ViolaDataSetup(object):
#######################################################
##### FINAL METHOD USED TO PRODUCE PATCHES FOR VIOLA
#######################################################
    @staticmethod
    def setup_negative_samples():
        '''Write file with info about location of negative examples
        '''
        output = open(utils.BG_FILE, 'w')
        for file in os.listdir(utils.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(utils.UNINHABITED_PATH+file+'\n')


    @staticmethod
    def setup_augmented_patches():
        '''
        No division between different roof sizes: if a roof has a size that is off, we resize it
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''
        in_path = utils.get_path(in_or_out=utils.IN, data_fold=utils.TRAINING)
        out_path = utils.get_path(viola=True, in_or_out=utils.IN, data_fold=utils.TRAINING)

        img_names_list = [img_name for img_name in os.listdir(in_path) if img_name.endswith('.jpg')]

        for roof_type in ['metal', 'thatch']:
            for img_id, img_name in enumerate(img_names_list):

                print 'Processing image: {0}'.format(img_name)
                img_path = in_path+img_name

                polygon_list = DataLoader.get_polygons(roof_type=roof_type, xml_name=img_name[:-3]+'xml', xml_path=in_path)
                roof_patches = DataLoader.extract_patches(polygon_list, img_path=img_path, grayscale=True)

                for roof_id, roof_img in enumerate(roof_patches):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)
                       
                    #if it's vertical, make it lie down
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    
                    #write basic positive example to the right folder
                    general_path = '{0}{1}_{2}_{3}'.format(out_path, roof_type, img_name[:-4], roof_id)

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
    def dat_file():
        '''
        Save .dat files containing location of training images
        One .dat file for metal and one for thatch
        '''
        in_path = utils.get_path(viola=True, in_or_out=utils.IN, data_fold=utils.TRAINING) 
        dat_file_name = dict()

        for roof_type in ['metal', 'thatch']:
            dat_file_name[roof_type] = '{0}.dat'.format(roof_type)

            try:
                dat_f = open(utils.DAT_PATH+dat_file_name[roof_type], 'w')

            except IOError as e:
                print e
                sys.exit(-1)

            else:
                img_names_list = [f for f in os.listdir(in_path) if (f.endswith('.jpg') and roof_type in f)]

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
                        dat_f.write(log_to_file)

            finally:
                dat_f.close()
        pdb.set_trace()

        return dat_file_name['metal'], dat_file_name['thatch']


    @staticmethod
    def vec_file_single(dat_file=''):
        '''Produce vec file for given dat file
        '''
        dat_path = '{0}'.format(utils.DAT_PATH+dat_file)
        sample_num = ViolaDataSetup.get_sample_num_from_dat(dat_path)
        w = raw_input('Width of {0}: '.format(dat_file[:-4]))
        h = raw_input('Height of {0}: '.format(dat_file[:-4]))

        vec_file = '{0}_num{3}_w{1}_h{2}.vec'.format(dat_file[:-4], w, h, sample_num)
        vec_path = '{0}{1}'.format(utils.VEC_PATH, vec_file)

        vec_cmd ='/usr/bin/opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_path, utils.BG_FILE, vec_path, sample_num, w, h)
        try:
            subprocess.check_call(vec_cmd, shell=True)
        except Exception as e:
            print e 
        return [(vec_file, sample_num, w, h)]


    @staticmethod
    def get_sample_num_from_dat(dat_path):
        with open(dat_path, 'rb') as csvfile:
            sample_num = 0
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                sample_num += int(row[1])
        return sample_num


########################################
## FINAL CODE USED BELOW
########################################

def main():
    ViolaDataSetup.setup_negative_samples() 
    ViolaDataSetup.setup_augmented_patches()
    thatch_dat, metal_dat = ViolaDataSetup.dat_file()

    ViolaDataSetup.vec_file_single(dat_file=thatch_dat)
    ViolaDataSetup.vec_file_single(dat_file=metal_dat)



if __name__ == '__main__':
    main()


