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

# TODO: depending on resolution step_size will be affected...
STEP_SIZE = settings.PATCH_H
OUTPUT_PATH = 'output'


class Pipeline(object):
    def __init__(self, step_size=None, test_files=None, test_folder=None, viola_process=True):
        self.step_size = step_size if step_size is not None else STEP_SIZE
        self.viola_process = viola_process

        #image_files
        if test_files is None:
            self.test_fnames = DataLoader.get_img_names_from_path(path = settings.INHABITED_PATH)
            self.test_img_path =  settings.INHABITED_PATH 
        else:
            self.test_fnames = test_files
            self.test_img_path = test_folder
        assert self.test_fnames is not None
        assert self.test_img_path is not None

        #OUTPUT FOLDER: create it if it doesn't exist
        #out_name = raw_input('Name of output folder: ')
        out_name = OUTPUT_PATH
        assert out_name != ''
        self.out_path = '../output/pipeline/{0}/'.format(out_name)
        if not os.path.isdir(self.out_path):
            subprocess.check_call('mkdir {0}'.format(self.out_path[:-1]), shell=True)
        print 'Will output files to: {0}'.format(self.out_path)

        #DETECTORS
        if self.viola_process:
            self.detector_paths = list()
            '''
            while True:
                cascade_name = raw_input('Cascade file to use: ' )
                detector = settings.CASCADE_PATH+cascade_name+'.xml'
                if cascade_name == '':
                    break
                self.detector_paths.append(detector)
            '''
            self.detector_path = DETECTORS
            print 'Using detectors: '+'\t'.join(self.detector_paths)

            #create report file
            self.report_path = self.out_path+'report.txt'
            with open(self.report_path, 'w') as report:
                report.write('\t'.join(self.detector_paths))

            self.viola = ViolaDetector(detector_paths=self.detector_paths, output_folder=self.out_path, save_imgs=True)
            #the output should be a picture with the windows marked!
        self.experiment = Experiment(preloaded=True, preloaded_path='conv5_nonroofs2_test20.0', flip=False, net_name='My_test_net', num_layers=5)

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
                self.image = np.asarray(self.image, dtype='float32')/255
                self.image_detections = cv2.imread(self.test_img_path+img_name)

            except IOError:
                print 'Cannot open '+self.test_img_path+img_name
                sys.exit(-1)

            self.image = np.transpose(self.image, (2,0,1))
            self.img_name = img_name

            if self.viola_process:
                rows, cols, _ = image.shape
                thatch_mask = np.zeros((rows, cols), dtype=bool)
                self.process_viola(rows, cols, verbose=True)

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


if __name__ == '__main__':
    #to detect a single image pass it in as a parameter
    #else, pipeline will use the files in settings.TEST_PATH folder
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:l:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    
    for opt, arg in opts:
        if opt == '-f':
            test_file = arg
            test_files = list()
            test_files.append(test_file)
        if opt == '-l':
            test_folder = arg

    pipe = Pipeline(test_files=None, test_folder=None, viola_process=False)
    #dictionaries accessed by file_name
    #pipe.experiment.test_preloaded()
    pipe.run()

