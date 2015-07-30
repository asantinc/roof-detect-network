import sys
import getopt
import os
import subprocess
import pdb
import re
from collections import defaultdict

import numpy as np
from scipy import misc
import cv2 

sys.path.append('~/roof/Lasagne/lasagne')
sys.path.append('~/roof/nolearn/nolearn')

import lasagne

#import convolution
import FlipBatchIterator as flip
import neural_network
from neural_network import Experiment, SaveLayerInfo, PrintLogSave
from viola_detector import ViolaDetector
import viola_detector_helpers
from my_net import MyNeuralNet
import viola_detector
import utils
from reporting import Detections, Evaluation

DEBUG = True

class Pipeline(object):
    def __init__(self,single_detector=True, in_path=None, out_path=None, neural=None, viola=None, pipe=None, out_folder_name=None):
        self.step_size = pipe['step_size'] if pipe['step_size'] is not None else utils.STEP_SIZE

        self.single_detector = single_detector
        self.in_path = in_path
        self.img_names = [img_name for img_name in os.listdir(self.in_path) if img_name.endswith('jpg')]
        self.out_path = out_path

        #create report file
        self.report_path = self.out_path+'report_pipe.txt'
        open(self.report_path, 'w').close()

        #Setup Viola 
        self.viola = ViolaDetector(pipeline=True, out_path=out_path, 
                                    in_path=in_path,
                                    folder_name=out_folder_name,
                                    save_imgs=True, **viola) 
        #Setup Neural network(s)
        if self.single_detector:
            self.net = Experiment(pipeline=True, **neural['metal'])
        else: 
            self.net = dict()
            self.net['metal'] = Experiment(pipeline=True, **neural['metal'])
            self.net['thatch'] = Experiment(pipeline=True, **neural['thatch'])
        
        #we keep track of detections before and after neural network 
        #so we can evaluate by how much the neural network is helping us improve
        #self.detections_before_neural = Detections()
        self.detections_after_neural = Detections()
        #self.evaluation_before_neural = Evaluation(detections=self.detections_before_neural, 
                                #method='pipeline', save_imgs=False, out_path=self.out_path, 
                                #report_name='before_neural.txt', folder_name=out_folder_name, 
                                #in_path=self.in_path, detector_names=viola['detector_names'])
        self.evaluation_after_neural = Evaluation(detections=self.detections_after_neural, 
                                method='pipeline', save_imgs=True, out_path=self.out_path,
                                report_name='after_neural.txt', folder_name=out_folder_name, 
                                in_path=self.in_path, detector_names=viola['detector_names'])

   
    def run(self):
        '''
        1. Find proposals using ViolaJones
        2. Resize the window and classify it
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        for i, img_name in enumerate(self.img_names):
            print '***************** Image {0}: {1}/{2} *****************'.format(img_name, i, len(self.img_names)-1)
            #VIOLA
            proposal_patches, proposal_coords, img_shape = self.find_viola_proposals(img_name=img_name)

            #NEURALNET
            classified_detections  = self.neural_classification(proposal_patches, proposal_coords)
            self.viola.evaluation.score_img(img_name, img_shape[:2])
            self.viola.evaluation.save_images(img_name, fname='beforeNeural')
            
            #set detections and score
            for roof_type in utils.ROOF_TYPES:
                self.detections_after_neural.set_detections(img_name=img_name, 
                                                    roof_type=roof_type, 
                                                    detection_list=classified_detections[roof_type])
            self.evaluation_after_neural.score_img(img_name, img_shape[:2])
            self.evaluation_after_neural.save_images(img_name, 'posNeural')

        self.evaluation_after_neural.print_report()

        #mark roofs on image
        #evaluate predictions
            #filter the thatched and metal roofs
            #compare the predictions made by viola and by viola+neural network


    def find_viola_proposals(self, img_name=None):
        '''Call viola to find coordinates of candidate roofs. 
        Extract those patches from the image, tranform them so they can be fed to neural network.
        Return both the coordinates and the patches.
        '''
        self.viola.detect_roofs(img_name=img_name)
        try:
            img_full = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            img_shape = img_full.shape
        except IOError as e:
            print e
            sys.exit(-1)

        if DEBUG:
            self.viola.evaluation.save_images(img_name)
        all_proposal_patches = dict()
        all_proposal_coords = dict()
        
        #extract patches for neural network classification
        for roof_type in ['metal', 'thatch']: 
            all_proposal_coords[roof_type] = self.viola.viola_detections.get_detections(img_name=img_name, roof_type=roof_type)
            patches = np.empty((len(all_proposal_coords[roof_type]), 3, utils.PATCH_W, utils.PATCH_H)) 

            for i, detection in enumerate(all_proposal_coords[roof_type]): 
                #extract the patch from the image using utils code
                img = utils.four_point_transform(img_full, detection)

                #transform the patch using utils code
                patch = utils.cv2_to_neural(img)
                patches[i, :, :,:] = patch 

            all_proposal_patches[roof_type] = patches  

        return all_proposal_patches, all_proposal_coords, img_shape


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
 

    def neural_classification(self, proposal_patches, proposal_coords):
        classified_detections = defaultdict(list)
        for roof_type in utils.ROOF_TYPES:
            #classify with neural network
            
            if proposal_patches[roof_type].shape[0] > 1:
                if self.single_detector: #we have a single net
                    classes = self.net.test(proposal_patches[roof_type])

                    #filter according to classification         
                    for detection, classification in zip(proposal_coords[roof_type], classes):
                        if classification == utils.NON_ROOF:
                            classified_detections['background'].append(detection)
                        elif classification == utils.METAL:
                            classified_detections['metal'].append(detection)
                        elif classification == utils.THATCH:
                            classified_detections['thatch'].append(detection)

                else: #we have one net per roof type
                    specific_net = self.net[roof_type]
                    classes = specific_net.test(proposal_patches[roof_type])
                     #filter according to classification         
                    for detection, classification in zip(proposal_coords[roof_type], classes):
                        if classification == 0:
                            classified_detections['background'].append(detection)
                        elif classification == 1:
                            classified_detections[roof_type].append(detection)
                        else:
                            raise ValueError('Unknown classification of patch')
            else:
                print 'No {0} detections'.format(roof_type)
        return classified_detections


   

    def save_img_detections(self, img_name, proposal_coords, predictions):
        raise ValueError('Incorrect method')
        img = cv2.imread(self.in_path+img_name)
        roofs = DataLoader().get_roofs(self.in_path+img_name[:-3]+'xml', img_name)
        for roof in roofs:
            cv2.rectangle(img, (roof.xmin, roof.ymin), (roof.xmin+roof.width, roof.ymin+roof.height), (0,255,0), 2)
        for (x,y,w,h), accept in zip(proposal_coords['metal'], predictions[img_name]['metal']):
            color = (0,0,0) if accept==1 else (0,0,255) 
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2) 
        cv2.imwrite(self.out_path+img_name, img)

def check_preloaded_paths_correct(preloaded_paths): 
    #ensure that if we have only one detector, that the network was trained with both types of roof
    #if we have more than one detector, ensure that each was trained on only one type of roof
    metal_network = None 
    thatch_network = None
    for i, path in enumerate(preloaded_paths):

        metal_num = (int(float(utils.get_param_value_from_filename(path, 'metal'))))
        thatch_num = (int(float(utils.get_param_value_from_filename(path,'thatch'))))
        nonroof_num = (int(float(utils.get_param_value_from_filename(path,'nonroof'))))

        assert nonroof_num > 0
        if len(preloaded_paths) == 1:
            assert metal_num > 0 and thatch_num > 0  
            metal_network = path
            thatch_network = path
        elif len(preloaded_paths) == 2:
            if (metal_num == 0 and thatch_num > 0):
                thatch_network = path
            if (metal_num > 0 and thatch_num == 0):
                metal_network = path
        else:
            raise ValueError('You passed in too many network weights to the pipeline.')

    #if we passed in two neural networks, ensure that we actually 
    #have one for metal and one for thatch    
    assert metal_network is not None and thatch_network is not None
    return metal_network, thatch_network


def setup_neural_viola_params(parameters, pipe_fname):
    '''
    Get parameters for the pipeline and the components of the pipeline:
    the neural network(s) and the viola detector(s)
    '''
    preloaded_paths = parameters['preloaded_path'].split() #there can be more than one network, separated by space
    single_detector_boolean = False if len(preloaded_paths) == 2 else True

    #PIPE PARAMS:the step size (probably not needed) and the neural nets
    metal_net, thatch_net = check_preloaded_paths_correct(preloaded_paths)
    pipe_params = dict() 
    preloaded_paths_dict = {'metal': metal_net, 'thatch': thatch_net}
    pipe_params = {'step_size':parameters['step_size'] ,'preloaded_paths': preloaded_paths_dict}

    neural_params = dict() #one set of parameters per roof type
    for roof_type, path in preloaded_paths_dict.iteritems():
        #NEURAL parameters: there could be two neural networks trained and used
        neural_param_num = int(utils.get_param_value_from_filename(path, 'params'))
        neural_params_fname = 'params{0}.csv'.format(neural_param_num) 

        params_path = '{0}{1}'.format(utils.get_path(params=True, neural=True), neural_params_fname)
        neural_params[roof_type] = neural_network.get_neural_training_params_from_file(params_path)
        neural_params[roof_type]['preloaded_path'] = path
        neural_params[roof_type]['net_name'] = path[:-len('.pickle')] 
        if single_detector_boolean:
            neural_params[roof_type]['roof_type'] = 'Both'

    #VIOLA PARAMS
    viola_params = dict()
    viola_data = neural_params['metal']['viola_data']
    combo_fname = 'combo{0}'.format(int(utils.get_param_value_from_filename(viola_data,'combo')))
    viola_params['detector_names'] = viola_detector_helpers.get_detectors(combo_fname)

    #get other non-default params
    possible_parameters = ['min_neighbors','scale', 'group', 'removeOff', 'rotate', 'mergeFalsePos'] 
    for param in possible_parameters:
        if param in viola_data:
            viola_params[param]= utils.get_param_value_from_filename(viola_data, param)

    return neural_params, viola_params, pipe_params, single_detector_boolean


def get_main_param_filenum():
    #get the pipeline number
    try:
        opts, args = getopt.getopt(sys.argv[1:], ":f")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    for opt in opts:
        if opt[0] == '-f':
            pipe_num = args[0]
    return pipe_num


if __name__ == '__main__':
    pipe_num = get_main_param_filenum()

    #get the parameters from the pipeline
    pipe_fname = 'pipe{}.csv'.format(pipe_num)
    parameters = utils.get_params_from_file( '{0}{1}'.format(utils.get_path(params=True, pipe=True), pipe_fname))
    neural_params, viola_params, pipe_params, single_detector_bool = setup_neural_viola_params(parameters, pipe_fname[:-len('.csv')])
    in_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.IN, pipe=True) 
    out_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.OUT, pipe=True, out_folder_name=pipe_fname[:-len('.csv')])   
    pipe = Pipeline(single_detector=single_detector_bool, in_path=in_path, out_path=out_path, 
                    pipe=pipe_params, neural=neural_params, viola=viola_params, out_folder_name=pipe_fname)
    pipe.run()


