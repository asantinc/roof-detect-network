from collections import OrderedDict
import os
import subprocess
from datetime import datetime
import sys
import pdb
import datetime
import csv


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

IN = 1  #input 
OUT = 2 #output

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
        overwrite = raw_input('Folder {0} exists; overwrite, y or n?'.format(out_folder_path))
        if overwrite == 'n':
            sys.exit(-1)
    print 'The following folder has been created: {0}'.format(out_folder_path)

def check_append(path_being_constructed, addition):
    path_being_constructed.append(addition)

    if os.path.isdir(''.join(path_being_constructed)):
        return path_being_constructed
    else:
        mkdir(out_folder_path=''.join(path_being_constructed), confirm=True) 


def get_path(in_or_out=None, out_folder_name=None, params=False, full_dataset=False, 
                    pipe=False, viola=False, template=False, neural=False, data_fold=None):
    '''
    Return path to either input, output or parameter files. If a folder does not exit, ask user for confirmation and create it
    '''
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
        assert in_or_out is not None
        if in_or_out == IN:
            #INPUT PATH
            if full_dataset==False:
                path.append('../data_original/')
            else:
                path.append('../data/')
            assert data_fold is not None
            if data_fold == TRAINING:
                path.append('training/')
                if neural:
                    path.append('(neural/')
                elif viola:
                    path.append('viola/')
                else:
                    path.append('source/')
            elif data_fold==VALIDATION:
                path.append('validation/')
            elif data_fold==TESTING:
                path.append('testing/')
            else:
                raise ValueError
        elif in_or_out == OUT:
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
        if out_folder_name is not None:
            check_append(path, out_folder_name)
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


def get_params_from_file(path):
    print 'Getting parameters from {0}'.format(path)
    parameters = dict()
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for par in reader:
            if len(par) == 0:
                break
            elif len(par)== 2:
                parameters[par[0].strip()] = par[1].strip()

            elif len(par) == 3:
                if par[0].strip() not in parameters:
                    parameters[par[0].strip()] = dict()
                parameters[par[0].strip()][par[1].strip()] = par[2].strip() 
            else:
                raise ValueError('Cannot process {0}'.format(path))
    return parameters


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, M


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

