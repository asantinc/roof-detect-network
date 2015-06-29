import os
import subprocess
import pdb
import math

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import get_data
from get_data import DataAugmentation
import experiment_settings as settings


class ViolaTrainer(object):
    @staticmethod
    def train_cascade(vec_info=None, stages=20, minHitRate=0.99999):
        cascades = list()
        for (vec_file, sample_num, w, h) in vec_info:
            print 'Training with vec file: {0}'.format(vec_file)
            cascade_folder = '../viola_jones/cascade_{0}/'.format(vec_file[:-4])
            cascades.append(cascade_folder+'cascade.xml')
            mkdir_cmd = 'mkdir {0}'.format(cascade_folder)
            try:
                subprocess.check_call(mkdir_cmd, shell=True)
            except Exception as e:
                print e
            cmd = list()
            cmd.append('opencv_traincascade')
            cmd.append('-data {0}'.format(cascade_folder))
            cmd.append('-vec ../viola_jones/vec_files/{0}'.format(vec_file))
            cmd.append('-bg ../viola_jones/bg.txt')
            cmd.append('-numStages {0}'.format(stages)) 
            cmd.append('-minHitRate {0}'.format(minHitRate))
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
    @staticmethod
    def setup_negative_samples():
        '''Write file with info about location of negative examples
        '''
        output = open(settings.BG_FILE, 'w')
        for file in os.listdir(settings.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(settings.UNINHABITED_PATH+file+'\n')


    @staticmethod
    def transform_roofs(padding=5):
        ''' Save augmented jpgs of data with padding, rotations and flips
        '''
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.INHABITED_PATH)
        roof_loader = get_data.DataLoader()

        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
            img_path = settings.INHABITED_PATH+img_name
    
            roofs, _, _ = roof_loader.get_roofs(xml_path)
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
                    general_path = '{0}{1}/img{2}_'.format(settings.VIOLA_AUGM_DATA, roof.roof_type, img_id)

                    for rotation in range(4):
                        if rotation == 0:
                            pass
                        elif rotation > 0:
                            roof_img = DataAugmentation.rotateImage(roof_img)
                        patch_path = '{0}id{1}_rot{2}'.format(general_path, roof_id, rotation)
                        cv2.imwrite('{0}.jpg'.format(patch_path), roof_img)
                        DataAugmentation.flip_save(roof_img, patch_path)


    @staticmethod
    def setup_positive_samples_full_image(padding=5):
        '''Save .dat files containing location of training images

        Each of the training images contains a single roof

        It processes four .dat files: three for metal roofs of different weight/height ratios, 
        and one for thatched roofs
        '''
        metal = [list() for i in range(3)]
        metal[0] = '{0}metal_{1}_tall_augment.dat'.format(settings.DAT_PATH, padding)
        metal[1] = '{0}metal_{1}_square_augment.dat'.format(settings.DAT_PATH, padding)
        metal[2] = '{0}metal_{1}_wide_augment.dat'.format(settings.DAT_PATH, padding)

        thatch_n = '{0}thatch_{1}_augment.dat'.format(settings.DAT_PATH, padding)

        with open(metal[0], 'w') as metal_f_tall, open(metal[1], 'w') as metal_f_square, open(metal[2], 'w') as metal_f_wide, open(thatch_n, 'w') as thatch_f:
            for roof_type in ['metal', 'thatch']:
                img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.VIOLA_AUGM_DATA+roof_type)
                for img_name in img_names_list:
                    img_path = settings.VIOLA_AUGM_DATA+roof_type+img_name
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
                        if roof_type == 'metal':
                            if ratio > 1.15:
                                metal_f_wide.write(log_to_file)
                            elif ratio < 0.80:
                                metal_f_tall.write(log_to_file)
                            else:
                                metal_f_square.write(log_to_file)
                        else:
                             thatch_f.write(log_to_file)
                return {'thatch':[thatch_n], 'metal': metal}
        

    @staticmethod
    def setup_positive_samples(padding=0, path=settings.INHABITED_PATH):
        '''
        Return .dat files containing positions and sizes of roofs in training images
        This uses the info about the Inhabited roofs but adds padding also and produces .Dat
        file with samples embedded in the full images
        '''
        metal = [list() for i in range(3)]
        metal[0] = '{0}metal_{1}_tall.dat'.format(settings.DAT_PATH, padding)
        metal[1] = '{0}metal_{1}_square.dat'.format(settings.DAT_PATH, padding)
        metal[2] = '{0}metal_{1}_wide.dat'.format(settings.DAT_PATH, padding)

        thatch_n = '{0}thatch_{1}_augment.dat'.format(settings.DAT_PATH, padding)

        with open(metal[0], 'w') as metal_f_tall, open(metal[1], 'w') as metal_f_square, open(metal[2], 'w') as metal_f_wide, open(thatch_n, 'w') as thatch_f:
            img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
            roof_loader = get_data.DataLoader()

            for img_name in img_names_list:
                metal = [list() for i in range(3)]
                print 'Processing image: {0}'.format(img_name)
                xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
                img_path = settings.INHABITED_PATH+img_name
        
                roofs, _, _ = roof_loader.get_roofs(xml_path)
                metal_log = [list() for i in range(3)]
                thatch_log = list()

                for roof in roofs:
                    #append roof characteristics separated by single space
                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, img_path)

                    if roof.roof_type == 'metal':
                        ViolaDataSetup.get_roof_dat(metal_log, roof)
                    elif roof.roof_type == 'thatch':
                        ViolaDataSetup.get_roof_dat(thatch_log, roof)

                if len(metal_log[0]) > 0:
                    metal_f_tall.write(ViolaDataSetup.get_dat_string(metal_log[0], img_path))
                if len(metal_log[1]) > 0:
                    metal_f_square.write(ViolaDataSetup.get_dat_string(metal_log[1], img_path))
                if len(metal_log[2]) > 0:
                    metal_f_wide.write(ViolaDataSetup.get_dat_string(metal_log[2], img_path))
                if len(thatch_log) > 0:
                    thatch_f.write(ViolaDataSetup.get_dat_string(thatch_log, img_path))


    @staticmethod
    def get_roof_dat(roof_lists, roof):
        '''Return string with roof position and size for .dat file

        Roof is added to list depending on width/height ratio
        '''
        string_to_add = '{0} {1} {2} {3}'.format(roof.xmin, roof.ymin, roof.width, roof.height)
        #if we care about roof size, add the roofs to the right list depending on their height/width ratio
        ratio = 0
        if roof.roof_type == 'metal':
            aspect_ratio = float(roof.width)/(roof.height)
            if (aspect_ratio > 1.5):                       #TALL ROOF
                roof_lists[2].append(string_to_add)
            elif aspect_ratio >= 0.75 and ratio <= 1.5:    #SQUARE
                roof_lists[1].append(string_to_add)
            elif aspect_ratio < 0.75:                      #WIDE ROOF
                roof_lists[0].append(string_to_add)
        else:
            roof_lists.append(string_to_add)


    @staticmethod
    def get_dat_string(roof_list, img_path):
        '''Return string formatted for .dat file
        '''
        return img_path+'\t'+str(len(roof_list))+'\t'+'\t'.join(roof_list)+'\n'


    @staticmethod
    def vec_file_samples():
        '''Save a vec file for every .dat file found in settings.DAT_PATH
        '''
        vec_info = list()

        for file_name in os.listdir(settings.DAT_PATH):
            if file_name.endswith('.dat'):
                vec_file_name = file_name[:-3]+'vec'
                vec_file = settings.VEC_PATH+vec_file_name
                sample_num = raw_input('Sample nums for {0}: '.format(file_name))
                w = raw_input('Width of {0}: '.format(file_name))
                h = raw_input('Height of {0}: '.format(file_name))

                vec_info.append( (vec_file_name, int(float(sample_num)), int(float(w)), int(float(h))) )

                dat_file = settings.DAT_PATH+file_name
                sample_cmd ='opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_file, settings.BG_FILE, vec_file, sample_num, w, h)
                try:
                    subprocess.check_call(sample_cmd, shell=True)
                    move_cmd = 'mv {0} ../viola_jones/all_dat/'.format(dat_file)
                    subprocess.check_call(move_cmd, shell=True)
                except Exception as e:
                    print e
        return vec_info


    @staticmethod
    def vec_file_single(dat_file=''):
        '''Produce vec file for given dat file
        '''
        vec_file = dat_file[:-4]+'.vec'
        vec_path = '{0}{1}'.format(settings.VEC_PATH, vec_file)
        dat_file = '{0}{1}'.format(settings.DAT_PATH, dat_file)
        sample_num = raw_input('Sample nums for {0}: '.format(dat_file[:-4]))
        w = raw_input('Width of {0}: '.format(dat_file[:-4]))
        h = raw_input('Height of {0}: '.format(dat_file[:-4]))

        vec_cmd ='opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_file, settings.BG_FILE, vec_path, sample_num, w, h)
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


