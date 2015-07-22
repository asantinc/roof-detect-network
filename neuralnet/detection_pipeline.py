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
from experiment_settings import Experiment, SaveLayerInfo, PrintLogSave 
from get_data import DataLoader, Roof
from viola_detector import ViolaDetector
import viola_detector_helpers
from my_net import MyNeuralNet
import viola_detector
import utils
from reporting import Detections, Evaluation

class Pipeline(object):
    def __init__(self, in_path=None, out_path=None, neural=None, viola=None, pipe=None, out_folder_name=None):
        self.step_size = pipe['step_size'] if pipe['step_size'] is not None else utils.STEP_SIZE

        self.in_path = in_path
        self.img_names = [img_name for img_name in os.listdir(self.in_path) if img_name.endswith('jpg')]
        self.out_path = out_path

        #create report file
        self.report_path = self.out_path+'report_pipe.txt'
        open(self.report_path, 'w').close()

        #Viola will spit out an evaluation of the viola detector to the pipe folder also
        if len(viola.keys()) > 0:
            self.viola = ViolaDetector(detector_names=viola['detectors'], out_path=out_path, 
                                    in_path=in_path,out_folder_name=out_folder_name,save_imgs=True) 
        else: 
            self.viola = None 
        
        self.experiment = Experiment(pipeline=True, **neural)

        self.detections_before_neural = Detections()
        self.detections_after_neural = Detections()
        self.evaluation_before_neural = Evaluation(detections=self.detections_before_neural, method='pipeline', save_imgs=False, out_path=self.out_path, 
                                        folder_name=out_folder_name, in_path=self.in_path, detector_names=viola['detectors'])
        self.evaluation_after_neural = Evaluation(detections=self.detections_after_neural, method='pipeline', save_imgs=True, out_path=self.out_path,
                                                         folder_name=out_folder_name, in_path=self.in_path, detector_names=viola['detectors'])

    def run_old(self):
        '''
        1. Find contours using Viola and Jones for candidate roofs
        2. Within each contour, use neural network to classify sliding patches
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        self.thatch_masks = dict()
        self.metal_masks = dict()
        self.roof_coords = dict()
        self.all_contours = dict()
        
        for img_name in list(self.in_path):
            '''
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
            '''
    
    def run(self):
        '''
        1. Find proposals using ViolaJones
        2. Within each contour, use neural network to classify sliding patches
            2.b actually, probably resize the window and classify it
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        for img_name in self.img_names:
            proposal_patches, proposal_coords = self.find_viola_proposals(img_name=img_name)
            metal_detections = list()
            thatch_detections = list()
            self.detections_before_neural.set_detections(img_name=img_name, detection_list=proposal_coords)  
            self.evaluation_before_neural.score_img(img_name)

            for roof_type in ['metal', 'thatch']: 
                #classify with net
                if proposal_patches[roof_type].shape[0] > 1:
                    temp_detections = self.experiment.test(proposal_patches[roof_type])
                    #filter according to classification         
                    for detection, roof_type in zip(proposal_coords[roof_type],temp_detections):
                        if roof_type == 0:
                            continue
                        elif roof_type == 1:
                            metal_detections.append(detection)
                        elif roof_type == 2:
                            thatch_detections.append(detection)
            self.detections_after_neural.set_detections(img_name=img_name, detection_list= {'metal':metal_detections, 'thatch': thatch_detections})  
            self.evaluation_after_neural.score_img(img_name)

        self.evaluation_before_neural.print_report()
        self.evaluation_after_neural.print_report()

        #mark roofs on image
        #evaluate predictions
            #filter the thatched and metal roofs
            #compare the predictions made by viola and by viola+neural network

    def save_img_detections(self, img_name, proposal_coords, predictions):
        img = cv2.imread(self.in_path+img_name)
        roofs = DataLoader().get_roofs(self.in_path+img_name[:-3]+'xml', img_name)
        for roof in roofs:
            cv2.rectangle(img, (roof.xmin, roof.ymin), (roof.xmin+roof.width, roof.ymin+roof.height), (0,255,0), 2)
        for (x,y,w,h), accept in zip(proposal_coords['metal'], predictions[img_name]['metal']):
            color = (0,0,0) if accept==1 else (0,0,255) 
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2) 
        cv2.imwrite(self.out_path+img_name, img)


    def find_viola_proposals(self, img_name=None, group=None, reject_levels=1.3, level_weights=5, scale=None, min_neighbors=1, eps=2):
        '''Call viola to find coordinated of candidate roofs. 
        Extract those patches from the image, tranform them so they can be fed to neural network.
        Return both the coordinates and the patches.
        '''
        #def detect_roofs(self, img_name=None, group=None, reject_levels=1.3, level_weights=5, scale=None, min_neighbors=1, eps=2):
        proposal_coords, img = self.viola.detect_roofs(img_name=img_name, group=group, reject_levels=reject_levels, 
                            level_weights=level_weights, scale=scale, min_neighbors=min_neighbors, eps=eps)

        #cv2 and lasagne deal with images differently
        img = np.asarray(img, dtype='float32')/255
        img = img.transpose(2, 0, 1)

        all_proposal_patches = dict()

        for roof_type in ['metal', 'thatch']: 
            patches = np.empty((len(proposal_coords[roof_type]), img.shape[0], utils.CROP_SIZE, utils.CROP_SIZE)) 

            for i, (x,y,w,h) in enumerate(proposal_coords[roof_type]):
                #resize it down to crop size so it can be fed to neural network
                patch = np.copy(img[:, y:y+h, x:x+w])               
                dst = np.empty((3, utils.CROP_SIZE, utils.CROP_SIZE), dtype='float32')
                dst[0,:,:] = cv2.resize(patch[0, :,:], (utils.CROP_SIZE, utils.CROP_SIZE), dst=dst[0,:,:], fx=0, fy=0, interpolation=cv2.INTER_AREA) 
                dst[1,:,:] = cv2.resize(patch[1, :,:], (utils.CROP_SIZE, utils.CROP_SIZE), dst=dst[1,:,:], fx=0, fy=0, interpolation=cv2.INTER_AREA) 
                dst[2,:,:] = cv2.resize(patch[2, :,:], (utils.CROP_SIZE, utils.CROP_SIZE), dst=dst[2,:,:], fx=0, fy=0, interpolation=cv2.INTER_AREA) 
                patches[i, :,:,:] = np.asarray(dst, dtype='float32')

            all_proposal_patches[roof_type] = patches
        return all_proposal_patches, proposal_coords



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
            vert_patches = ((h - utils.PATCH_H) // self.step_size) + 1  #vert_patches = h/utils.PATCH_H

            #get patches along roof height
            for vertical in range(vert_patches):
                y_pos = y+(vertical*self.step_size)
                #along roof's width
                self.classify_horizontal_patches((x,y,w,h), y_pos=y_pos)

            #get patches from the last row also
            if (h % utils.PATCH_H>0) and (h > utils.PATCH_H):
                leftover = h-(vert_patches*utils.PATCH_H)
                y_pos = y_pos-leftover
                self.classify_horizontal_patches((x,y,w,h), y_pos=y_pos)



    def classify_horizontal_patches(self, patch=None, y_pos=-1):
        '''Get patches along the width of a patch for a given y_pos (i.e. a given height in the image)
        '''
        #roof_type = utils.METAL if roof.roof_type=='metal' else utils.THATCH

        x,y,w,h = patch
        hor_patches = ((w - utils.PATCH_W) // self.step_size) + 1  #hor_patches = w/utils.PATCH_W

        for horizontal in range(hor_patches):
            
            #get cropped patch
            x_pos = x+(horizontal*self.step_size)
            full_patch = self.image[:, y_pos:y_pos+utils.PATCH_H, x_pos:x_pos+utils.PATCH_W]
            full_patch = self.experiment.scaler.transform2(full_patch)

            diff = (utils.PATCH_W-utils.CROP_SIZE)/2
            candidate = full_patch[:, diff:diff+utils.CROP_SIZE, diff:diff+utils.CROP_SIZE]
           
            if candidate.shape != (3,32,32):
                print 'ERROR: patch too small, cannot do detection\n'
                continue
            #do network detection, add additional singleton dimension
            prediction = self.experiment.net.predict(candidate[None, :,:,:])
            if prediction[0] != utils.NON_ROOF:
                if prediction[0] == utils.METAL:
                    color = (255,255,255)
                elif prediction[0] == utils.THATCH:
                    color = (0,0,255)
                cv2.rectangle(self.image_detections, (x_pos+4, y_pos+4), (x_pos+utils.PATCH_H-4, y_pos+utils.PATCH_H-4), color, 1)



def setup_neural_viola_params(parameters):
    pipe_params = parameters['pipe']

    #set up neural parameters
    neural_params = parameters['neural']
    neural_params['num_layers'] = int(float(neural_params['preloaded_path'][4:5]))
    neural_params['net_name'] = neural_params['preloaded_path'][:-(len('.pickle'))]+'_metal'+parameters['viola']['metal_combo']+'_thatch'+parameters['viola']['thatch_combo']+'/'

    #set up viola parameters
    viola_params = dict()
    detectors = dict()
    detectors['metal'] = viola_detector_helpers.get_detectors(parameters['viola']['metal_combo'])['metal']
    detectors['thatch'] = viola_detector_helpers.get_detectors(parameters['viola']['thatch_combo'])['thatch']   
    viola_params['detectors'] = detectors
    viola_params['group'] = parameters['viola']['group']

    return neural_params, viola_params, pipe_params


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], ":f")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    print opts, args
    for opt in opts:
        if opt[0] == '-f':
            pipe_num = args[0]

    parameters = utils.get_params_from_file('{0}pipe{1}.csv'.format(utils.get_path(params=True, pipe=True), pipe_num))
    print parameters
    neural_params, viola_params, pipe_params = setup_neural_viola_params(parameters)

    in_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.IN, pipe=True) 
    out_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.OUT, pipe=True, out_folder_name=neural_params['net_name'])   

    pipe = Pipeline(in_path=in_path, out_path=out_path, pipe=pipe_params, neural=neural_params, viola=viola_params, out_folder_name=neural_params['net_name'])
    pipe.run()


