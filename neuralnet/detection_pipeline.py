import sys
import getopt
import os
import subprocess
import pdb
import re

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
        pdb.set_trace()
        if len(viola.keys()) > 0:
            self.viola = ViolaDetector(out_path=out_path, in_path=in_path,
                                        folder_name=out_folder_name,
                                        save_imgs=True, **viola) 
        else: 
            self.viola = None 
        self.experiment = Experiment(pipeline=True, **neural)

        #we keep track of detections before and after neural network so we can evaluate by how much the neural network is helping us improve
        self.detections_before_neural = Detections()
        self.detections_after_neural = Detections()
        self.evaluation_before_neural = Evaluation(detections=self.detections_before_neural, method='pipeline', save_imgs=False, out_path=self.out_path, 
                                report_name='before_neural.txt', folder_name=out_folder_name, in_path=self.in_path, detector_names=viola['detectors'])
        self.evaluation_after_neural = Evaluation(detections=self.detections_after_neural, method='pipeline', save_imgs=True, out_path=self.out_path,
                                report_name='after_neural.txt', folder_name=out_folder_name, in_path=self.in_path, detector_names=viola['detectors'])

   
    def run(self):
        '''
        1. Find proposals using ViolaJones
        2. Resize the window and classify it
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        for i, img_name in enumerate(self.img_names):
            print '***************** Image {0}: {1}/{2} *****************'.format(img_name, i, len(self.img_names))
            proposal_patches, proposal_coords = self.find_viola_proposals(img_name=img_name)
            self.evaluation_before_neural.score_img(img_name)

            metal_detections = list()
            thatch_detections = list()
            for roof_type in ['metal', 'thatch']: 
                #classify with net
                proposals = proposal_patches[roof_type]

                #classify with neural network
                if proposal_patches[roof_type].shape[0] > 1:
                    neural_classification = self.experiment.test(proposals)

                    #filter according to classification         
                    for detection, roof_type in zip(proposal_coords[roof_type], neural_classification):
                        if roof_type == utils.NON_ROOF:
                            continue
                        elif roof_type == utils.METAL:
                            metal_detections.append(detection)
                        elif roof_type == utils.THATCH:
                            thatch_detections.append(detection)
                else:
                    print 'No {0} detections'.format(roof_type)
            self.detections_after_neural.set_detections(img_name=img_name, roof_type='metal', detection_list=metal_detections)
            self.detections_after_neural.set_detections(img_name=img_name, roof_type='thatch', detection_list=thatch_detections)  
            self.evaluation_after_neural.score_img(img_name)

        self.evaluation_before_neural.print_report()
        self.evaluation_after_neural.print_report()

        #mark roofs on image
        #evaluate predictions
            #filter the thatched and metal roofs
            #compare the predictions made by viola and by viola+neural network


    def find_viola_proposals(self, img_name=None):
        '''Call viola to find coordinated of candidate roofs. 
        Extract those patches from the image, tranform them so they can be fed to neural network.
        Return both the coordinates and the patches.
        '''
        img = self.viola.detect_roofs(img_name=img_name)
        all_proposal_patches = dict()
        all_proposal_coords = dict()

        for roof_type in ['metal', 'thatch']: 
            all_proposal_coords[roof_type] = self.viola.viola_detections.get_detections(img_name=img_name, roof_type=roof_type)
            self.detections_before_neural.set_detections(img_name=img_name, roof_type=roof_type, detection_list=all_proposal_coords[roof_type])  

            patches = np.empty((len(all_proposal_coords[roof_type]), 3, utils.PATCH_W, utils.PATCH_H)) 

            for i, detection in enumerate(all_proposal_coords[roof_type]): 
                #extract the patch from the image using utils code
                img = utils.four_point_transform(img, detection)
                #transform the patch using utils code
                patches[i, :, :,:] = utils.cv2_to_neural(img)

            all_proposal_patches[roof_type] = patches  
        return all_proposal_patches, all_proposal_coords



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



def setup_neural_viola_params(parameters):
    '''
    Pipe file contains info about

    '''
    #PIPE PARAMS
    pipe_params = {'step_size':parameters['step_size'] ,'preloaded_path': parameters['preloaded_path']}

    #NEURAL parameters
    start_params = parameters['preloaded_path'].index('params')
    end_index = start_params+re.search('_', parameters['preloaded_path'][start_params:]).start()
    print start_params, end_index
    neural_params_file = '{0}.csv'.format(parameters['preloaded_path'][start_params:end_index])
    params_path = '{0}{1}'.format(utils.get_path(params=True, neural=True), neural_params_file)
    neural_params = neural_network.get_neural_training_params_from_file(params_path)
    neural_params['net_name'] = parameters['preloaded_path'][:-(len('.pickle'))]

    #VIOLA PARAMS
    viola_params = dict()
    viola_data = neural_params['viola_data']
    end_combo = re.search('_', viola_data).start()
    combo_fname = viola_data[:end_combo]
    viola_params['detector_names'] = viola_detector_helpers.get_detectors(combo_fname)
    #get other non-default params
    possible_parameters = ['min_neighbors','scale', 'group', 'removeOff', 'rotate', 'mergeFalsePos'] 
    for param in possible_parameters:
        if param in viola_data:
            start_index = re.search(param, viola_data).start() 
            param_value_start = start_index+len(param)
           
            end_index_re = re.search('_', viola_data[param_value_start:])
            if end_index_re is not None:
                end_index = param_value_start + end_index_re.start()
                value = viola_data[param_value_start:end_index]
            else:
                value = viola_data[param_value_start:]

            try:#try to conver to a float, if not will keep as string
                float_value = float(value)
                value = float_value
            except:
                continue
            finally:
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                viola_params[param] = value
    return neural_params, viola_params, pipe_params


if __name__ == '__main__':
    #get the pipeline number
    try:
        opts, args = getopt.getopt(sys.argv[1:], ":f")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    print opts, args
    for opt in opts:
        if opt[0] == '-f':
            pipe_num = args[0]
    #get the parameters from the pipeline
    parameters = utils.get_params_from_file('{0}pipe{1}.csv'.format(utils.get_path(params=True, pipe=True), pipe_num))
    neural_params, viola_params, pipe_params = setup_neural_viola_params(parameters)

    in_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.IN, pipe=True) 
    out_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.OUT, pipe=True, out_folder_name=neural_params['net_name'])   
    print in_path
    pipe = Pipeline(in_path=in_path, out_path=out_path, pipe=pipe_params, neural=neural_params, viola=viola_params, out_folder_name=neural_params['net_name'])
    pipe.run()


