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


class ViolaTrainer(object):
    @staticmethod
    def train_cascade(vec_files=None, feature_type='haar', max_false_alarm_rate=0.5, stages=20, minHitRate=0.99999, roof_type=None, padding=-1):
        cascades = list()
        roof_type = roof_type
        if vec_files is None:    
            vec_files = get_data.DataLoader().get_img_names_from_path(path=settings.VEC_PATH, extension='.vec') 
        for vec_file in vec_files:
            vec_type = vec_file[:5] if roof_type == 'metal' else vec_file[:6]
            if (roof_type is None or vec_type  == roof_type):
                #get cascade parameters from file name
                w = vec_file[-10:-8]
                h = vec_file[-6:-4]
                index = vec_file.find('num') 
                assert index != -1
                sample_num = vec_file[index+3:-12]
                assert int(float(sample_num)) > 0

                print 'Training with vec file: {0}'.format(vec_file)
                cascade_folder = '../viola_jones/cascade_{0}_FA{1}_{2}/'.format(vec_file[:-4], max_false_alarm_rate, feature_type)
                cascades.append(cascade_folder+'cascade.xml')
                mkdir_cmd = 'mkdir {0}'.format(cascade_folder)
                try:
                    subprocess.check_call(mkdir_cmd, shell=True)
                except Exception as e:
                    print e
                
                cmd = list()
                cmd.append('/usr/bin/opencv_traincascade')
                cmd.append('-data {0}'.format(cascade_folder))
                cmd.append('-vec ../viola_jones/vec_files/{0}'.format(vec_file))
                cmd.append('-bg ../viola_jones/bg.txt')
                cmd.append('-numStages {0}'.format(stages)) 
                cmd.append('-minHitRate {0}'.format(minHitRate))
                if feature_type != 'haar':
                    cmd.append('-featureType LBP')
                cmd.append('-maxFalseAlarmRate {0}'.format(max_false_alarm_rate))
                cmd.append('-precalcValBufSize 1024 -precalcIdxBufSize 1024')
                numPos = int(float(sample_num)*.8)
                cmd.append('-numPos {0} -numNeg {1}'.format(numPos, numPos*2))
                cmd.append('-w {0} -h {1}'.format(w, h))
                train_cmd = ' '.join(cmd)
                try:
                    print train_cmd
                    subprocess.check_call(train_cmd, shell=True)
                except Exception as e:
                    print e
        return cascades


class ViolaDataSetup(object):


#######################################################
##### FINAL METHOD USED TO PRODUCE PATCHES FOR VIOLA
#######################################################
    @staticmethod
    def final_cascades_data_setup(equalized=True, path=settings.TRAINING_PATH, padding=0):
        '''
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
        roof_loader = get_data.DataLoader()
        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            xml_path = path+img_name[:-3]+'xml'
            img_path = path+img_name
    
            roofs = roof_loader.get_roofs(xml_path, img_name)
            metal_log = list()
            thatch_log = list()
            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if equalized:
                    gray = cv2.equalizeHist(gray)
            except IOError:
                print 'Cannot open '+img_path
            else:
                for roof_id, roof in enumerate(roofs):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)

                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, padding, img_path)
                    roof_type = roof.roof_type 
                    roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])
                       
                    #if it's vertical, make it lie down
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    
                    #we want to split data depending on its width/height ratio
                    roof_shape = 'square'
                    if roof_img.shape[0] < 0.70*roof_img.shape[1] or roof_img.shape[0]*.70 > roof_img.shape[1]:
                        roof_shape = 'rectangular'
 
                    #write basic positive example to the right folder
                    img_name_short = img_name[:-4]
                    equalize_folder = 'equalized' if equalized else 'not_equalized'
                    general_path = '{1}/{2}/{3}/{4}_{5}'.format(settings.TRAINING_VIOLA_POS_PATH, 
                                                                roof_type, roof_shape, equalize_folder, img_name_short, roof_id, padding)
                    cv2.imwrite('{0}{1}.jpg'.format(settings.TRAINING_VIOLA_POS_PATH, general_path), roof_img)    
                    cv2.imwrite('{0}{1}.jpg'.format(settings.TRAINING_VIOLA_POS_AUGM_FULL_PATH, general_path), roof_img)

                    #calculate and write the augmented images too
                    DataAugmentation.flip_pad_save(img_path, roof, settings.TRAINING_VIOLA_POS_AUGM_FULL_PATH+general_path, equalize=equalized)



    @staticmethod
    def original_dataset_setup_single_size_patches_augment(in_path=settings.ORIGINAL_TRAINING_PATH, out_path=settings.ORIGINAL_VIOLA_PATCHES_AUGM):
        '''
        No division between different roof sizes: if a roof has a size that is off, we resize it
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''

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
    def final_cascades_data_setup_single_size(in_path=settings.TRAINING_PATH, original_dataset=False, out_path=None,  padding=0):
        '''
        No division between different roof sizes: if a roof has a size that is off, we resize it
        Make them lie down, save patches to folder
        Augment patches, save them to augmented folder
        '''
        img_names_list = get_data.DataLoader().get_img_names_from_path(in_path=path)
        roof_loader = get_data.DataLoader()
        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            xml_path = path+img_name[:-3]+'xml'
            img_path = path+img_name
    
            roofs = roof_loader.get_roofs(xml_path, img_name)
            metal_log = list()
            thatch_log = list()
            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if equalized:
                    gray = cv2.equalizeHist(gray)
            except IOError:
                print 'Cannot open '+img_path
            else:
                for roof_id, roof in enumerate(roofs):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)

                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, padding, img_path)
                    roof_type = roof.roof_type 
                    roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])
                       
                    #if it's vertical, make it lie down
                    if roof_img.shape[0] > roof_img.shape[1]:
                        roof_img = DataAugmentation.rotateImage(roof_img, clockwise=True)
                    
                    #we want to split data depending on its width/height ratio
                    roof_shape = 'square'
                    if roof_img.shape[0] < 0.70*roof_img.shape[1] or roof_img.shape[0]*.70 > roof_img.shape[1]:
                        roof_shape = 'rectangular'
 
                    #write basic positive example to the right folder
                    img_name_short = img_name[:-4]
                    equalize_folder = 'equalized' if equalized else 'not_equalized'
                    general_path = '{1}/{2}/{3}/{4}_{5}'.format(settings.TRAINING_VIOLA_POS_PATH, 
                                                                roof_type, roof_shape, equalize_folder, img_name_short, roof_id, padding)
                    cv2.imwrite('{0}{1}.jpg'.format(settings.TRAINING_VIOLA_POS_PATH, general_path), roof_img)    
                    cv2.imwrite('{0}{1}.jpg'.format(settings.TRAINING_VIOLA_POS_AUGM_FULL_PATH, general_path), roof_img)

                    #calculate and write the augmented images too
                    pdb.set_trace()
                    DataAugmentation.flip_pad_save(img_path, roof, settings.TRAINING_VIOLA_POS_AUGM_FULL_PATH+general_path, equalize=equalized)


    @staticmethod
    def setup_negative_samples():
        '''Write file with info about location of negative examples
        '''
        output = open(settings.BG_FILE, 'w')
        for file in os.listdir(settings.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(settings.UNINHABITED_PATH+file+'\n')


    @staticmethod
    def transform_roofs(padding=5, path=settings.TRAINING_PATH, out_path= settings.VIOLA_AUGM_DATA):
        ''' Save augmented jpgs of data with padding, rotations and flips
        '''
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
        roof_loader = get_data.DataLoader()

        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            xml_path = path+img_name[:-3]+'xml'
            img_path = path+img_name
    
            roofs = roof_loader.get_roofs(xml_path, img_name)
            metal_log = list()
            thatch_log = list()

            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except IOError:
                print 'Cannot open '+img_path
            else:
                for roof_id, roof in enumerate(roofs):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)

                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, padding, img_path)

                    roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])
                    general_path = '{0}{1}/img{2}_'.format(out_path, roof.roof_type, img_id)

                    for rotation in range(4):
                        if rotation == 0:
                            pass
                        elif rotation > 0:
                            roof_img = DataAugmentation.rotateImage(roof_img)
                        patch_path = '{0}id{1}_rot{2}'.format(general_path, roof_id, rotation)
                        cv2.imwrite('{0}.jpg'.format(patch_path), roof_img)
                        DataAugmentation.flip_save(roof_img, patch_path)


    @staticmethod
    def dat_file_setup_positive_samples_full_image_single_size(in_path=settings.VIOLA_AUGM_DATA, dat_file_name=None):
        '''Save .dat files containing location of training images
        Each of the training images contains a single roof
        '''
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
    def setup_positive_samples_full_image(padding=5, size_divide=True, in_path=settings.VIOLA_AUGM_DATA):
        '''Save .dat files containing location of training images

        Each of the training images contains a single roof

        It processes four .dat files: three for metal roofs of different weight/height ratios, 
        and one for thatched roofs
        '''
        if size_divide:
            metal = [list() for i in range(3)]
            metal[0] = '{0}metal_{1}_tall.dat'.format(settings.DAT_PATH, padding)
            metal[1] = '{0}metal_{1}_square.dat'.format(settings.DAT_PATH, padding)
            metal[2] = '{0}metal_{1}_wide.dat'.format(settings.DAT_PATH, padding)
        else:
            metal_n =  '{0}metal_{1}.dat'.format(settings.DAT_PATH, padding) 
        thatch_n = '{0}thatch_{1}.dat'.format(settings.DAT_PATH, padding)

        try:
            if size_divide:
                metal_f_tall = open(metal[0], 'w')
                metal_f_square = open(metal[1], 'w')
                metal_f_wide = open(metal[2], 'w')
            else:
                metal_f = open(metal_n, 'w')
            thatch_f = open(thatch_n, 'w')
        except IOError as e:
            print e
            sys.exit(-1)
        else:
            for roof_type in ['metal', 'thatch']:
                img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.VIOLA_AUGM_DATA+roof_type)
                for img_name in img_names_list:
                    img_path = settings.VIOLA_AUGM_DATA+roof_type+'/'+img_name
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
                        ratio = float(width)/height
                        if roof_type == 'metal' and size_divide:
                            if ratio > 1.15:
                                metal_f_wide.write(log_to_file)
                            elif ratio < 0.80:
                                metal_f_tall.write(log_to_file)
                            else:
                                metal_f_square.write(log_to_file)
                        elif roof_type == 'metal':
                            metal_f.write(log_to_file)
                        else:
                            thatch_f.write(log_to_file)

        finally:
            if size_divide:
                metal_f_tall.close()
                metal_f_square.close()
                metal_f_wide.close()
            else:
                metal_f.close()
            thatch_f.close() 



    @staticmethod
    def setup_positive_samples(padding=0, path=settings.TRAINING_PATH, size_divide=True):
        '''
        Return .dat files containing positions and sizes of roofs in training images
        This uses the info about the Inhabited roofs but adds padding also and produces .Dat
        file with samples embedded in the full images. Because we use the whole image, we can't
        do data augmentation.
        '''
        all_dat_names = list() #return this with info about dat files
        if size_divide:
            metal = [list() for i in range(3)]
            metal[0] = '{0}metal_{1}_tall_augm0.dat'.format(settings.DAT_PATH, padding)
            metal[1] = '{0}metal_{1}_square_augm0.dat'.format(settings.DAT_PATH, padding)
            metal[2] = '{0}metal_{1}_wide_augm0.dat'.format(settings.DAT_PATH, padding)
            all_dat_names.extend([m for m in metal])
        else:
            metal_n =  '{0}metal_{1}_augm0.dat'.format(settings.DAT_PATH, padding) 
            all_dat_names.append(metal_n)
        thatch_n = '{0}thatch_{1}_augm0.dat'.format(settings.DAT_PATH, padding)
        all_dat_names.append(thatch_n)

        try:
            if size_divide:
                metal_f_tall = open(metal[0], 'w')
                metal_f_square = open(metal[1], 'w')
                metal_f_wide = open(metal[2], 'w')
            else:
                metal_f = open(metal_n, 'w')
            thatch_f = open(thatch_n, 'w')
        except IOError as e:
            print e
            sys.exit(-1)
        else:
            img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
            roof_loader = get_data.DataLoader()

            for img_name in img_names_list:
                print 'Processing image: {0}'.format(img_name)
                xml_path = path+img_name[:-3]+'xml'
                img_path = '../'+path+img_name
        
                roofs = roof_loader.get_roofs(xml_path, img_name)
                metal_log = [list() for i in range(3)] if size_divide else list()
                thatch_log = list()

                for roof in roofs:
                    #append roof characteristics separated by single space
                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, img_path)

                    if roof.roof_type == 'metal':
                        ViolaDataSetup.get_roof_dat(metal_log, roof, size_divide=size_divide)
                    elif roof.roof_type == 'thatch':
                        ViolaDataSetup.get_roof_dat(thatch_log, roof)

                if size_divide:
                    if len(metal_log[0]) > 0:
                        metal_f_tall.write(ViolaDataSetup.get_dat_string(metal_log[0], img_path))
                    if len(metal_log[1]) > 0:
                        metal_f_square.write(ViolaDataSetup.get_dat_string(metal_log[1], img_path))
                    if len(metal_log[2]) > 0:
                        metal_f_wide.write(ViolaDataSetup.get_dat_string(metal_log[2], img_path))
                else:
                    metal_f.write(ViolaDataSetup.get_dat_string(metal_log, img_path))

                if len(thatch_log) > 0:
                    thatch_f.write(ViolaDataSetup.get_dat_string(thatch_log, img_path))

        finally:
            if size_divide:
                metal_f_tall.close()
                metal_f_square.close()
                metal_f_wide.close()
            else:
                metal_f.close()
            thatch_f.close() 

        return all_dat_names #return the dat file names created


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


class ViolaBasicTrainer(object):
    @staticmethod
    def produce_imgs_simple_detector():
        '''TODO: this may not be needed 
        Squares and rectangles to use as simple detectors instead of the roof images
        '''
        path = '../viola_jones/data/data_simple/'

        background = np.zeros((200,120))
        background[10:190, 10:110] = 1
        blur_back = cv2.GaussianBlur(background, (21,21), 0)
        misc.imsave('{0}simple_vert_rect.jpg'.format(path), blur_back)

        background = np.zeros((120,200))
        background[10:110, 10:190] = 1
        blur_back = cv2.GaussianBlur(background, (21,21), 0)
        misc.imsave('{0}simple_horiz_rect.jpg'.format(path), blur_back)

        rotated_blur = ndimage.rotate(blur_back, 45)
        misc.imsave('{0}simple_diag_rect.jpg'.format(path), rotated_blur)

        #square
        square = np.zeros((200,200))
        square[10:190, 10:190] = 1
        blur_square = cv2.GaussianBlur(square, (21,21), 0)
        misc.imsave('{0}simple_square.jpg'.format(path), blur_square)

        rot_square = ndimage.rotate(blur_square, 45)
        misc.imsave('{0}simple_diag_square.jpg'.format(path), rot_square)


    @staticmethod
    def setup_positive_samples_simple_detector(roof_type='metal', padding=5):
        ''' Create .dat files from jpgs, where the samples are just a repetition of a single jpg
        '''
        path = '../viola_jones/data/data_simple/'
        out_path = '../viola_jones/all_dat/'
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
        
        #produce repeated samples of each .jpg
        for img_name in img_names_list:
            print 'Processing {0}'.format(img_name)
            img_path = path+img_name
            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                out_file = open(out_path+img_name[:-3]+'dat', 'w')
            except IOError:
                print 'Cannot do IO'
            else:
                height, width, _ = img.shape
                for i in range(100):
                    #append roof characteristics separated by single space
                    position_string = '{0} {1} {2} {3}'.format(0, 0, width, height)
                    log_to_file = '{0}\t{1}\t{2}\n'.format('../../'+img_path, 1, position_string)
                    out_file.write(log_to_file)

########################################
## FINAL CODE USED BELOW
########################################

def train_cascade(max_false_alarm=0.2, feature_type='LBP'):
    #TRAINING CASCADE
    no_details = True
    roof_type = ''  
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:t:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            v = arg if arg.endswith('.vec') else arg+'.vec'
            no_details = False
        elif opt == '-t':
            roof_type = arg
    roof_type = 'metal' if v[:1]=='m' else 'thatch'

    if no_details:
        v = raw_input('Enter vec file: ')
        t = raw_input('Type of roof: ' )
        roof_type = 'metal' if t=='m' else 'thatch'
    vecs = [v]
    ViolaTrainer.train_cascade(vec_files=vecs, max_false_alarm_rate=max_false_alarm, feature_type=feature_type, roof_type=roof_type)



def data_setup():
    ViolaDataSetup.setup_negative_samples() 
    dat_names = ViolaDataSetup.setup_positive_samples(padding=0, path=settings.TRAINING_PATH, size_divide=False)     
    ViolaDataSetup.vec_file_samples()

def produce_all_viola_patches():
    ViolaDataSetup.final_cascades_data_setup(equalized=True)

def produce_dat_files():
    for roof_type in ['metal', 'thatch']:
        for shape in ['rectangular', 'square']:
            for illum in ['equalized', 'not_equalized']:
                for aug in [True, False]:
                    augmented_path = settings.TRAINING_VIOLA_POS_AUGM_PATH if aug else settings.TRAINING_VIOLA_POS_PATH
                    in_path = '{0}{1}/{2}/{3}/'.format(augmented_path, roof_type, shape, illum)
                    augmented = 'augm1' if aug else 'augm0'
                    dat_file = '{0}_{1}_{2}_{3}.dat'.format(roof_type, shape, illum, augmented)
                    ViolaDataSetup.dat_file_setup_positive_samples_full_image_single_size(in_path=in_path, dat_file_name= dat_file)

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
    #produce_dat_files()
    #ViolaDataSetup.vec_file_samples()
    #produce_all_viola_patches()
#    ViolaDataSetup.original_dataset_setup_single_size_patches_augment()
#    raise ValueError('Remember to set the right out_path and in_path for patches, and in_path for .dat files')

    #SET UP viola positive patches from original dataset
    #ViolaDataSetup.original_dataset_setup_single_size_patches_augment()
    #in_path = settings.ORIGINAL_VIOLA_PATCHES_AUGM
    #for roof_type in ['metal', 'thatch']:
    #    dat_file = '{0}_original_singlesize_equalized_augm1.dat'.format(roof_type)
    #    ViolaDataSetup.dat_file_setup_positive_samples_full_image_single_size(in_path=in_path+roof_type+'/', dat_file_name= dat_file)


#    ViolaDataSetup.vec_file_single('metal_original_singlesize_equalized_augm1.dat')
#    ViolaDataSetup.vec_file_single('metal_original_singlesize_equalized_augm1.dat')
#    ViolaDataSetup.vec_file_single('thatch_original_singlesize_equalized_augm1.dat')

    train_cascade(max_false_alarm=0.4, feature_type='LBP')
