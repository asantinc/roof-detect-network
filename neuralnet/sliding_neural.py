import sys
import getopt
import os
import subprocess
import pdb
import re
from collections import defaultdict
import pickle

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
from my_net import MyNeuralNet
import utils
from reporting import Detections, Evaluation
from timer import Timer
import suppression

DEBUG = True

class SlidingNeural(object):
    def __init__(self, groupThres=0, groupBounds=False, erosion=0, suppress=False, overlapThresh=0.3,
                    single_detector=True, 
                    in_path=None, out_path=None, neural=None, 
                    pipe=None, out_folder_name=None):
        
        '''

        Parameters:
        ------------------
        groupThres bool
            Decides if we should do grouping on neural detections
        '''

        self.groupThreshold = int(groupThres)
        self.groupBounds = groupBounds
        self.erosion = erosion
        self.suppress = suppress
        self.overlapThresh = overlapThresh

        self.single_detector = single_detector
        self.in_path = in_path
        self.img_names = [img_name for img_name in os.listdir(self.in_path) if img_name.endswith('jpg')]
        self.out_path = out_path
               
        #Setup Neural network(s)
        if self.single_detector:#you should train a single network
            self.net = Experiment(pipeline=True, **neural['metal'])
        else: 
            self.net = dict()
            self.net['metal'] = Experiment(pipeline=True, **neural['metal'])
            self.net['thatch'] = Experiment(pipeline=True, **neural['thatch'])

        self.evaluation = Evaluation(detections=self.detections_after_neural, 
                                method='sliding', save_imgs=True, out_path=self.out_path,
                                folder_name=out_folder_name, 
                                in_path=self.in_path)

   
    def run(self):
        '''
        1. Sliding window proposals
        2. Resize the window and classify it
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        neural_time = 0
        for i, img_name in enumerate(self.img_names):
            print '***************** Image {0}: {1}/{2} *****************'.format(img_name, i, len(self.img_names)-1)

            #NEURALNET
            with Timer() as t:
                classified_detections  = self.neural_classification(proposal_patches, proposal_coords) 
                #set detections and score
                for roof_type in utils.ROOF_TYPES:
                    if self.groupThreshold > 0 and roof_type == 'metal':
                        #need to covert to rectangles
                        boxes = utils.get_bounding_boxes(np.array(classified_detections[roof_type]))
                        grouped_boxes, weights = cv2.groupRectangles(np.array(boxes).tolist(), self.groupThreshold)
                        classified_detections[roof_type] = utils.convert_detections_to_polygons(grouped_boxes) 
                        #convert back to polygons

                    elif self.groupBounds and roof_type == 'metal':
                        #grouping with the minimal bound of all overlapping rects
                        classified_detections[roof_type] = self.group_min_bound(classified_detections[roof_type], img_shape[:2], erosion=self.erosion)

                    elif self.suppress and roof_type == 'metal':
                        #proper non max suppression from Felzenszwalb et al.
                        classified_detections[roof_type] = self.non_max_suppression(classified_detections[roof_type])

                    self.detections_after_neural.set_detections(img_name=img_name, 
                                                        roof_type=roof_type, 
                                                        detection_list=classified_detections[roof_type])
            neural_time += t.secs 

            self.evaluation.score_img(img_name, img_shape[:2], contours=self.groupBounds)
            self.evaluation.save_images(img_name, 'posNeural')
               self.evaluation_after_neural.detections.total_time = (neural_time)
        self.evaluation_after_neural.print_report(print_header=False, stage='neural')
        

    def non_max_suppression(self,polygons):
        #we start with polygons, get the bounding box of it
        rects = utils.get_bounding_boxes(np.array(polygons))
        #covert the bounding box to what's requested by the non_max_suppression
        boxes = utils.rects2boxes(rects)
        boxes_suppressed = suppression.non_max_suppression(boxes, overlapThresh=self.overlapThresh)
        polygons_suppressed = utils.boxes2polygons(boxes_suppressed)
        return polygons_suppressed 


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




    def detect_roofs_folder(self, path):
        pass


    def detect_roofs(self, image):
        # loop over the image pyramid
        for resized in utils.pyramid(image, scale=1.5):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in utils.sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                # WINDOW

                # since we do not have a classifier, we'll just draw the window
                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)

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

    #PIPE PARAMS:the step size (probably not needed) and the neural nets to be used
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
    groupThres = 0
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:g:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    for opt, arg in opts:
        if opt == '-f':
            pipe_num = arg
        elif opt == '-g':
            groupThres = int(float(arg))
    return pipe_num, groupThres



if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:m:")
    except getopt.GetoptError:
        sys.exit(2)
        print 'Command line failed'
    for opt, arg in opts:
        if opt == '-s':
            scale = arg
        if opt == '-m':
            min_size = (int(float(arg)), int(float(arg)))

    #need to get an image!
    image = cv2.imread('../data_original/training/source/0002.jpg')
    (winW, winH) = (128, 128)
    # loop over the image pyramid

