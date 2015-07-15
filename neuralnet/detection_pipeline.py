import sys
import getopt
import os
import subprocess
import pdb

import numpy as np
from scipy import misc
import cv2 

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

import lasagne

#import convolution
import FlipBatchIterator as flip
import experiment_settings as settings
from experiment_settings import Experiment, SaveLayerInfo, PrintLogSave 
from get_data import DataLoader, Roof
from viola_detector import ViolaDetector
from my_net import MyNeuralNet
import viola_detector


class Pipeline(object):
    def __init__(self, detectors=None, out_folder_name=None, validation=True, step_size=None, neural=None):
        self.detector_paths = detectors
        self.step_size = step_size if step_size is not None else settings.STEP_SIZE
        self.setup_input_output_report(validation, out_folder_name)

        #Viola detection
        self.viola=None
        if detectors is not None:
            self.viola = ViolaDetector(detector_names=detectors, out_path=self.out_path, folder_name=out_folder_name, save_imgs=False)

        self.experiment = Experiment(**neural)


    def setup_input_output_report(self, validation, out_folder_name):
        '''
        Create output folder if it doesn't exist
        Create report file, write to it the detector info
        '''
        #INPUT PATH and FILES
        self.test_img_path = settings.VALIDATION_PATH if validation else settings.TESTING_PATH 
        self.test_fnames = DataLoader.get_img_names_from_path(path =self.test_img_path  )
        
        #OUTPUT FOLDER
        self.out_path ='{0}{1}/'.format(settings.PIPE_OUT, out_folder_name)
        if not os.path.isdir(self.out_path):
            subprocess.check_call('mkdir {0}'.format(self.out_path), shell=True)
        else:
            print 'Directory {0} already exists \n'.format(self.out_path)
        print 'Will output files to: {0}'.format(self.out_path)

        #create report file
        self.report_path = self.out_path+'report.txt'
        with open(self.report_path, 'w') as report:
            report.write('\t'.join(self.detector_paths))



    def run(self):
        '''
        1. Find contours using Viola and Jones for candidate roofs
        2. Within each contour, use neural network to classify sliding patches
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        self.thatch_masks = dict()
        self.metal_masks = dict()
        self.roof_coords = dict()
        self.all_contours = dict()
        
        for img_name in list(self.test_fnames):
            print 'Pre-processing image: {0}'.format(img_name)
            try:
                self.image = cv2.imread(self.test_img_path+img_name)
                self.image = np.asarray(self.image, dtype='float32')
                self.image_detections = cv2.imread(self.test_img_path+img_name)

            except IOError:
                print 'Cannot open '+self.test_img_path+img_name
                sys.exit(-1)

            self.image = np.transpose(self.image, (2,0,1))
            self.img_name = img_name

            if self.detectors:
                rows, cols, _ = image.shape
                thatch_mask = np.zeros((rows, cols), dtype=bool)
                self.process_viola(rows, cols, verbose=True)
                raise ValueError('Classfy patches with network')
            else:
                self.sliding_convolution()
            
            cv2.imwrite(self.out_path+self.img_name+'test.jpg', self.image_detections)     

    
    def process_viola(self, rows, cols, img_path=None, verbose=False):
        #Find candidate roof contours using Viola for all types of roof
        #returns list with as many lists of detections as the detectors we have passed
        self.viola.detect_roofs(img_name=self.img_name, img_path=self.test_img_path+self.img_name)
        print 'Detected {0} candidate roofs'.format(len(self.viola.roofs_detected[self.img_name]))
        if verbose:
            self.viola.mark_detections_on_img(img=self.image, img_name=self.img_name)

        #get the mask and the contours for the detections
        detection_mask, _ = self.viola.get_patch_mask(img_name=self.img_name, rows=rows, cols=cols)
        patch_location = self.out_path+self.img_name+'_mask.jpg'
        misc.imsave(patch_location, detection_mask)

        self.all_contours[self.img_name] = self.viola.get_detection_contours(patch_location, self.img_name)
    

    def sliding_convolution(self):
        '''
        Classify patches of an image using neural network. If viola is True, the patches are fed from ViolaJones 
        algorithm. Otherwise, the entire image is processed
        '''
        print '********* PREDICTION STARTED *********\n'
        bound_rects = list()
        if self.viola_process:
            #for each contour, do a sliding window detection
            contours = self.all_contours[self.img_name]
            for cont in contours:
                bound_rects.append(cv2.boundingRect(cont))
        else:
            c, rows, cols = self.image.shape
            bound_rects.append((0,0,cols,rows))

        for x,y,w,h in bound_rects:
            vert_patches = ((h - settings.PATCH_H) // self.step_size) + 1  #vert_patches = h/settings.PATCH_H

            #get patches along roof height
            for vertical in range(vert_patches):
                y_pos = y+(vertical*self.step_size)
                #along roof's width
                self.classify_horizontal_patches((x,y,w,h), y_pos=y_pos)

            #get patches from the last row also
            if (h % settings.PATCH_H>0) and (h > settings.PATCH_H):
                leftover = h-(vert_patches*settings.PATCH_H)
                y_pos = y_pos-leftover
                self.classify_horizontal_patches((x,y,w,h), y_pos=y_pos)



    def classify_horizontal_patches(self, patch=None, y_pos=-1):
        '''Get patches along the width of a patch for a given y_pos (i.e. a given height in the image)
        '''
        #roof_type = settings.METAL if roof.roof_type=='metal' else settings.THATCH

        x,y,w,h = patch
        hor_patches = ((w - settings.PATCH_W) // self.step_size) + 1  #hor_patches = w/settings.PATCH_W

        for horizontal in range(hor_patches):
            
            #get cropped patch
            x_pos = x+(horizontal*self.step_size)
            full_patch = self.image[:, y_pos:y_pos+settings.PATCH_H, x_pos:x_pos+settings.PATCH_W]
            full_patch = self.experiment.scaler.transform2(full_patch)

            diff = (settings.PATCH_W-settings.CROP_SIZE)/2
            candidate = full_patch[:, diff:diff+settings.CROP_SIZE, diff:diff+settings.CROP_SIZE]
           
            if candidate.shape != (3,32,32):
                print 'ERROR: patch too small, cannot do detection\n'
                continue
            #do network detection, add additional singleton dimension
            prediction = self.experiment.net.predict(candidate[None, :,:,:])
            if prediction[0] != settings.NON_ROOF:
                if prediction[0] == settings.METAL:
                    color = (255,255,255)
                elif prediction[0] == settings.THATCH:
                    color = (0,0,255)
                cv2.rectangle(self.image_detections, (x_pos+4, y_pos+4), (x_pos+settings.PATCH_H-4, y_pos+settings.PATCH_H-4), color, 1)


def get_params_from_file(file_name):
    parameters = dict()
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for par in reader:
            if len(par) == 2:
                parameteres[par[0].strip()] = par[1].strip()
    return parameters


if __name__ == '__main__':
    '''
    param_file = 'params'+raw_input('Enter param file number :')+'.csv'
    params = get_params_from_file(settings.PIPE_PARAMS_PATH+param_file) 

    Pipeline()
    if params['net_name'] == 0:
        params['net_name'] = time_stamped(param_file)
        print 'Network name is: {0}'.format(params['net_name'])
    if params['roofs_only']:
        params['net_name'] = params['net_name']+'_roofs'

    #to detect a single image pass it in as a parameter
    #else, pipeline will use the files in settings.TEST_PATH folder
    '''
    #preloaded=, preloaded_path=neural_name, flip=False, net_name='My_test_net', num_layers=5

    weight_name = 'conv5_nonroofs2_test20.0'
    neural_params={'preloaded':True, 'preloaded_path':weight_name, 'flip':False, 'net_name':'test1', 'num_layers':5}
    #get params 
    try:
        opts, args = getopt.getopt(sys.argv[1:], "v:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    no_viola = False
    for opt, arg in opts:
        if opt == '-n':
            no_viola = True
    if no_viola:
        detectors, combo_f_name = None, 'no_viola'
    else:
        detectors, combo_num =viola_detector.get_detectors()
        combo_f_name = 'combo{0}'.format(combo_num)

    pipe = Pipeline(detectors=detectors, neural=neural_params, out_folder_name=combo_f_name)
    pipe.run()


