from collections import OrderedDict
import os
import subprocess
from datetime import datetime
import sys
import pdb
import datetime


###################################
#types of roof
###################################
NON_ROOF = 0
METAL = 1
THATCH = 2
#Constants for image size
IMG_SIZE = 40
CROP_SIZE = 32
PATCH_W = PATCH_H = 40

TRAINING = 1
VALIDATION = 2
TESTING = 3
SMALL_TEST = 4

###################################
#VIOLA TRAINING
###################################
#Viola constants
RECTIFIED_COORDINATES = '../data/source/bounding_inhabited/'
BG_FILE = '../viola_jones/bg.txt'
DAT_PATH = '../viola_jones/all_dat/'
VEC_PATH = '../viola_jones/vec_files/'
CASCADE_PATH = '../viola_jones/cascades/'


###################################
#NEURAL TRAINING PATCHES
###################################
TRAINING_NEURAL_POS = '../data/training/neural/positives/'  #where the true positives from training source are 
TRAINING_NEURAL_NEG = '../data/training/neural/negatives/'  

#Constants for training neural network
#NET_PARAMS_PATH = "../parameters/net_params/"
OUT_REPORT = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/neural/report_'  
OUT_HISTORY = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/neural/history_' 
OUT_IMAGES = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/neural/images_'

FTRAIN = '../data/training/'
FTRAIN_LABEL = '../data/training/labels.csv'
#TEST_PATH = '../data/test/'


###################################
#Constants for debugging
###################################
VERBOSITY = 1   #varies from 1 to 3
DEBUG = False

###################################
#Pipeline
###################################
STEP_SIZE = PATCH_H #used by detection pipeline



def time_stamped(fname=''):
    if fname != '':
        fmt = '{fname}_%m-%d-%H-%M/'
    else:
        fmt =  '%m-%d-%H-%M/'
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

def mkdir(out_folder_path=None, confirm=False):
    assert out_folder_path is not None
    if not os.path.isdir(out_folder_path):
        if confirm:
            answer = raw_input('{0} does not exist. Do you want to create it? Type "y" or "n"'.format(out_folder_path))
            if answer == 'n':
                sys.exit(-1)
        subprocess.check_call('mkdir {0}'.format(out_folder_path), shell=True)
    else:
        overwrite = raw_input('Folder exists; overwrite, y or n?')
        if overwrite == 'n':
            sys.exit(-1)
    print 'The following folder has been created: {0}'.format(out_folder_path)

def check_append(path_being_constructed, addition):
    path_being_constructed.append(addition)

    if os.path.isdir(''.join(path_being_constructed)):
        return path_being_constructed
    else:
        mkdir(out_folder_path=''.join(path_being_constructed), confirm=True) 


def get_path(input_or_output=None, params=False, full_dataset=False, 
                    pipe=False, viola=False, template=False, neural=False, data_fold=None):
    path = list()
    if params:
        #PARAMETER FILES
        check_append(path, '../parameters/')
        if neural:
            check_append(path, 'net_params/') 
        elif pipe:
            check_append(path, 'pipe_params/')
        elif viola:
            check_append(path, 'detector_combos/')
        else:
            raise ValueError('There are no parameter files for templating')
    else:
        assert input_or_output is not None
        if input_or_output == 'input':
            #INPUT PATH
            if full_dataset==False:
                check_append(path,'../data_original/')
            else:
                check_append(path,'../data/')
            assert data_fold is not None
            if data_fold == TRAINING:
                check_append(path,'training/')
                if neural:
                    check_append(path,'(neural/')
                elif viola:
                    check_append(path,'viola/')
                else:
                    check_append(path,'source/')
            elif data_fold==VALIDATION:
                check_append(path,'validation/')
            elif data_fold==TESTING:
                check_append(path,'testing/')
            else:
                raise ValueError
        elif input_or_output == 'output':
            #OUTPUT FOLDERS
            check_append(path,'/afs/inf.ed.ac.uk/group/ANC/s0839470/output/')
            if viola:
                check_append(path,'viola/')
            elif template:
                check_append(path,'templating/')
            elif neural:
                check_append(path,'neural/')
            elif pipe:
                check_append(path,'pipe/')
            else:
                raise ValueError('Cannot create path. Try setting viola, pipe, template or neural to True to get a path')
            
            original = 'original_' if full_dataset == False else ''
            if data_fold==TRAINING:
                check_append(path,'with_{0}training_set/'.format(original))
            elif data_fold==TESTING:
                check_append(path,'with_{0}testing_set/'.format(original))
            elif data_fold==VALIDATION:
                check_append(path,'with_{0}validation_set/'.format(original))
            elif data_fold==SMALL_TEST:
                check_append(path,'with_small_test/')
            else:
                raise ValueError('Even though you requested the path of an output folder, you have to specify if you created the output from training, validation or testing data')
        else:
            raise ValueError('Must specify input or output folder.')
    return ''.join(path)
     

def print_debug(to_print, verbosity=1):
    #Print depending on verbosity level
    if verbosity <= VERBOSITY:
        print str(to_print)

def get_img_size(path):
    for i, f_name in enumerate(os.listdir(path)):
        img = cv2.imread(path+f_name)
        print img.shape[0], img.shape[1]
        
        if i%20==0:
            pdb.set_trace()
