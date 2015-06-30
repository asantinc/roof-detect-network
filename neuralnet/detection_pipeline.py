import sys
import getopt
import os
import subprocess

import numpy as np
from scipy import misc

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
STEP_SIZE = settings.PATCH_H / 4
OUTPUT_PATH = 'output'
#TODO: fill out detectors
DETECTORS = [settings.CASCADE_PATH++'.xml', '']

class Pipeline(object):
    def __init__(self, step_size=None, test_files=None, test_folder=None):
        self.step_size = step_size if step_size is not None else STEP_SIZE

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
        if not os.path.isdir(out_name):
            subprocess.check_call('mkdir {0}'.format(self.out_path[:-1]), shell=True)
        print 'Will output files to: {0}'.format(self.out_path)

        #DETECTORS
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
        self.neural_net = self.setup_neural_net(net_name='conv5_nonroofs1_test20_roofs', num_layers=5)


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
        
        for img_name in self.test_fnames:
            print 'Pre-processing image: {0}'.format(img_name)
            try:
                image = misc.imread(self.test_img_path+img_name)
            except IOError:
                print 'Cannot open '+self.test_img_path+img_name
                sys.exit(-1)
            rows, cols, _ = image.shape
            
            thatch_mask = np.zeros((rows, cols), dtype=bool)

            self.process_viola(img_name, image, rows, cols, verbose=True, img_path=self.test_img_path)

            metal_coords, thatch_coords = self.sliding_convolution(image, img_name)
            roofs_detected[img_name] = (metal_coords, thatch_coords)


    def process_viola(self, img_name, image, rows, cols, img_path=None, verbose=False):
        '''
        Find candidate roof contours using Viola for all types of roof
        '''
        #returns list with as many lists of detections as the detectors we have passed
        self.viola.detect_roofs(img_name=img_name, img_path=self.test_img_path+img_name)
        print 'Detected {0} candidate roofs'.format(len(self.viola.roofs_detected[img_name]))
        path = img_path if img_path is not None else settings.TEST_PATH+img_name
        if verbose:
            self.viola.mark_detections_on_img(img=image, img_name=img_name)

        #get the mask and the contours for the detections
        detection_mask, _ = self.viola.get_patch_mask(img_name=img_name, rows=rows, cols=cols)
        patch_location = self.out_path+img_name+'_mask.jpg'
        misc.imsave(patch_location, detection_mask)

        self.all_contours[img_name] = self.viola.get_detection_contours(patch_location, img_name)


    def sliding_convolution(self, img, img_name):
        #for each contour, do a sliding window detection
        contours = self.all_contours[img_name]
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            vert_patches = ((h - settings.PATCH_H) // self.step_size) + 1  #vert_patches = h/settings.PATCH_H

            #get patches along roof height
            for vertical in range(vert_patches):
                y_pos = roof.ymin+(vertical*self.step_size)
                #along roof's width
                self.classify_horizontal_patches((x,y,w,h), img=img, y_pos=y_pos)

            #get patches from the last row also
            if (h % settings.PATCH_W>0) and (h > settings.PATCH_H):
                leftover = h-(vert_patches*settings.PATCH_H)
                y_pos = y_pos-leftover
                self.classify_horizontal_patches((x,y,w,h), img=img, y_pos=y_pos)

        return metal_coords, thatch_coords


    def classify_horizontal_patches(self, patch=None, img=None, roof=None, y_pos=-1):
        '''Get patches along the width of a patch for a given y_pos (i.e. a given height in the image)
        '''
        roof_type = settings.METAL if roof.roof_type=='metal' else settings.THATCH

        h, w = roof.get_roof_size()
        hor_patches = ((w - settings.PATCH_W) // self.step_size) + 1  #hor_patches = w/settings.PATCH_W

        for horizontal in range(hor_patches):
            x_pos = roof.xmin+(horizontal*self.step_size)
            candidate = img[x_pos:x_pos+settings.PATCH_W, y_pos:y_pos+settings.PATCH_H]
            #do network detection
            prediction = self.experiment.test_preloaded_single(candidate)
            pdb.set_trace()
    
    
    def setup_neural_net(self, net_name='', num_layers=0):
        '''
        Parameters:
        net_name: name of network
        '''
        #TODO: the net name should have info about number of layers
        layers = MyNeuralNet.produce_layers(num_layers)      
        printer = PrintLogSave()
        net = MyNeuralNet(
            layers=layers,
            input_shape=(None, 3, settings.CROP_SIZE, settings.CROP_SIZE),
            output_num_units=3,
            
            output_nonlinearity=lasagne.nonlinearities.softmax,
            preproc_scaler = None, 
            
            #learning rates
            update_learning_rate=0.01,
            update_momentum=0.9,
            
            #printing
            net_name=net_name,
            on_epoch_finished=[printer],
            on_training_started=[SaveLayerInfo()],

            #data augmentation
            batch_iterator_test=flip.CropOnlyBatchIterator(batch_size=128),
            batch_iterator_train=flip.FlipBatchIterator(batch_size=128),
            
            max_epochs=-1,
            verbose=1,
            )  
        self.experiment = Experiment(net=net, printer=printer)

if __name__ == '__main__':
    #to detect a single image pass it in as a parameter
    #else, pipeline will use the files in settings.TEST_PATH folder
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:l:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2) 
    
    test_files = None
    test_folder = None
    for opt, arg in opts:
        if opt == '-f':
            test_file = arg
            test_files = list()
            test_files.append(test_file)
        if opt == '-l':
            test_folder = arg
    
    pipe = Pipeline(test_files=test_files, test_folder=test_folder)
    #dictionaries accessed by file_name
    metal_detections, thatch_detections = pipe.run()



