import sys
import getopt
import os
import subprocess
import pdb
import re
from collections import defaultdict
import pickle
import cProfile

import numpy as np
from scipy import misc
import cv2 
from collections import namedtuple

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
from timer import Timer
import suppression
from slide_neural import SlidingWindowNeural
from ensemble import Ensemble
from auc_curve import AucCurve

DEBUG = False 

class Pipeline(object):
    def __init__(self, method=None,
                    full_dataset=True,
                    metal_groupThres=0.1, thatch_groupThres=0.1, 
                    groupBounds=False, erosion=0, suppress=None, 
                    pickle_viola=None,# single_detector=True, 
                    in_path=None, out_path=None, neural=None, 
                    ensemble=None, 
                    detector_params=None, pipe=None, out_folder_name=None, net_threshold=0.5):
        '''
        Parameters:
        ------------------
        groupThres bool
            Decides if we should do grouping on neural detections
        method:string
            Can be either 'viola' or 'sliding_window'
        '''
        assert method=='viola' or method=='slide'

        self.method = method
        self.full_dataset = full_dataset

        self.groupThres = dict()
        self.groupThres['thatch'] = metal_groupThres
        self.groupThres['metal'] = thatch_groupThres
        self.groupBounds = groupBounds
        self.erosion = erosion

        #self.single_detector = single_detector
        self.in_path = in_path
        self.img_names = [img_name for img_name in os.listdir(self.in_path) if img_name.endswith('jpg')]
        self.img_names = self.img_names if DEBUG==False else self.img_names[:1]
        self.out_path = out_path
        
        #Setup Viola: if we are given an evaluation directly, don't bother running viola 
        if self.method == 'viola':
            self.pickle_viola = pickle_viola
            if self.pickle_viola is None:
                self.viola = ViolaDetector(pipeline=True, out_path=out_path, 
                                            in_path=in_path,
                                            folder_name=out_folder_name,
                                            save_imgs=True, **detector_params) 
            else:
                with open(pickle_viola, 'rb') as f:
                    self.viola_evaluation = pickle.load(f) 
                    self.viola_evaluation.in_path = self.in_path
                    self.viola_evaluation.out_path = self.out_path


        #Setup the sliding window
        elif self.method == 'slide':
            self.slider = SlidingWindowNeural(full_dataset=self.full_dataset, in_path=self.in_path, out_path=self.out_path, **detector_params) 
        else:
            raise ValueError('Need to specific either viola or sliding window')

        self.ensemble = ensemble

        #EVALUATION OBJECTS
        self.auc_thresholds = [.5]
        self.detections_after_neural = list()
        self.evaluation_after_neural = list()
        detector_names = detector_params['detector_names'] if self.method=='viola' else None
        for thres in self.auc_thresholds:
            detections = Detections()
            self.detections_after_neural.append(detections)
            self.evaluation_after_neural.append(Evaluation(detections=detections, 
                                        method='pipeline', save_imgs=True, out_path=self.out_path,
                                        auc_threshold=thres, folder_name=out_folder_name, in_path=self.in_path, detector_names=detector_names))

        self.auc = AucCurve(self.img_names, self.evaluation_after_neural[0].correct_roofs, self.out_path, self.method)
        print self.img_names

    def run(self):
        '''
        1. Find proposals using ViolaJones or sliding window
        2. Resize the window and classify it
        3. Net returns a list of the roof coordinates of each type - saved in roof_coords
        '''
        neural_time = 0
        viola_time = 0
        self.img_names = self.img_names[:15] if DEBUG else self.img_names
        for i, img_name in enumerate(self.img_names):
            print '***************** Image {0}: {1}/{2} *****************'.format(img_name, i, len(self.img_names)-1)

            #VIOLA
            rect_detections = dict()
            if self.method == 'viola':
                if self.pickle_viola is None:
                    img = self.viola.detect_roofs(img_name=img_name)
                    current_viola_detections = self.viola.viola_detections 
                    viola_time = self.viola.evaluation.detections.total_time
                else:#use the pickled detections for speed in testing the neural network
                    current_viola_detections = self.viola_evaluation.detections
                    viola_time = self.viola_evaluation.detections.total_time 
                proposal_patches, proposal_coords, img_shape = self.find_viola_proposals(current_viola_detections, img_name=img_name)
                for roof_type in utils.ROOF_TYPES:
                    rect_detections[roof_type] = utils.polygons2boxes(proposal_coords[roof_type])

                        #SLIDING WINDOW
            elif self.method == 'slide':
                with Timer() as t:
                    #get the roofs with sliding detector
                    proposal_coords, rect_detections = self.slider.get_windows(img_name) 
                    #convert them to patches
                    proposal_patches, img_shape = self.find_slider_proposals(rect_detections, img_name=img_name)
                print 'Sliding window detection for one image took {} seconds'.format(t.secs)
            else:
                print 'Unknown detection method {}'.format(self.method)
                sys.exit(-1)

            '''
            with open('combo11.pickle', 'wb') as f:
                pickle.dump(self.viola, f)
            sys.exit()
            '''

            #self.auc.set_detections(rect_detections, img_name)
            self.print_detections(rect_detections, img_name, '_viola')
           
            #NEURALNET
            print 'Starting neural classification of image {}'.format(img_name)
            with Timer() as t:
                #NOTE: classified detections only has roofs with prob >= 0.5
                classified_detections, probs  = self.neural_classification_AUC(proposal_patches, rect_detections) 
            print 'Classification took {} secs'.format(t.secs)
            neural_time += t.secs

            #GROUPING
            rect_detections, probs, grouping_time  = self.nonmax_suppression(rect_detections, probs)   
            neural_time += grouping_time
            
            self.print_detections({'metal':classified_detections['metal'][0],'thatch':classified_detections['thatch'][0]}, img_name, '_neural')


            det = dict()
            for roof_type in utils.ROOF_TYPES:
                det[roof_type] = rect_detections[roof_type][probs[roof_type]>0.5]
            self.print_detections(det, img_name, '_grouped')

            #AUC USING THE GROUPED DETECTIONS
            self.auc.set_detections(rect_detections, img_name)
            self.auc.set_probs(probs, img_name)

            #SCORING THE CURRENT IMAGE
            fast_scoring = True# False
            #if self.full_dataset:
            #    fast_scoring = True           
            for t, thres in enumerate(self.auc_thresholds):
                for roof_type in utils.ROOF_TYPES:
                    detections = classified_detections[roof_type][t]
                    print 'NEURAL SCORING FOR THRESHOLD {}'.format(thres)
                    self.detections_after_neural[t].set_detections(img_name=img_name, 
                                                            roof_type=roof_type, 
                                                            detection_list=detections)
                self.evaluation_after_neural[t].score_img(img_name, img_shape[:2], fast_scoring=fast_scoring, contours=self.groupBounds)

        
            #self.print_detections({'metal':classified_detections['metal'][0],'thatch':classified_detections['thatch'][0] }, img_name, '_neural0.5')

        #FINAL EVALUATION
        if self.method == 'viola': 
            if self.pickle_viola is None:
                self.viola.evaluation.print_report(print_header=True, stage='viola')
            else:
                self.viola_evaluation.print_report(print_header=True, stage='viola')
        for t, thres in enumerate(self.auc_thresholds):
            self.evaluation_after_neural[t].detections.total_time = neural_time+viola_time
            header = False if self.method=='viola' else True
            self.evaluation_after_neural[t].print_report(print_header=header, stage='neural', report_name='report_thres{}.txt'.format(thres))
        

    def print_detections(self, detections, img_name, title):
        if detections is not None:
            for roof_type, detects in detections.iteritems():
                img = cv2.imread(self.in_path+img_name)

                utils.draw_detections(self.evaluation_after_neural[0].correct_roofs[roof_type][img_name], img, rects=True, color=(0,255,0), thickness=6)
                utils.draw_detections(detects, img, rects=True, color=(255,0,0), thickness=3)

                cv2.imwrite('debug/{}_{}_{}.jpg'.format(self.groupThres[roof_type], img_name[:-4],roof_type, title), img)


    def nonmax_suppression(self, rect_detections, probs):
        with Timer() as t:
            #set detections and score
            for roof_type in utils.ROOF_TYPES:
                #proper non max suppression from Felzenszwalb et al.
                rect_detections[roof_type], probs[roof_type] = suppression.non_max_suppression(rect_detections[roof_type], probs[roof_type], overlapThres = self.groupThres[roof_type])
        print 'Grouping took {} seconds'.format(t.secs)
        return rect_detections, probs, t.secs



    def get_correct_class_per_detection(self,rect_detections, img_name): 
        #this is needed to build the Recall precision curve

        #get the best class guess of the detections by scoring it with ground truth
        self.slider.detections.set_detections(roof_type='thatch', detection_list=rect_detections['thatch'], img_name=img_name)
        self.slider.detections.set_detections(roof_type='metal', detection_list=rect_detections['metal'], img_name=img_name)
        #score the image
        self.slider.evaluation.score_img(img_name=img_name, img_shape=(-1,-1), fast_scoring=True) #since we use fast scoring, we don't need the img_shape
        
        #get the proper class by looking at the best score for each detection
        correct_classes = dict()
        for roof_type in utils.ROOF_TYPES:
            correct_classes[roof_type] = np.zeros((len(rect_detections[roof_type]))) 
            for d, (detection, score) in enumerate(self.slider.detections.best_score_per_detection[img_name][roof_type]):
                correct_classes[roof_type][d] = 0 if score<0.5 else 1
            correct_classes[roof_type] = list(correct_classes[roof_type])
        return correct_classes


    def group_min_bound(self, polygons, img_shape, erosion=0):
        '''
        Attempt at finding the minbound of all overlapping rects and merging them
        to a single detection. This unfortunately will merge nearby roofs.
        '''
        bitmap = np.zeros(img_shape, dtype='uint8')
        utils.draw_detections(np.array(polygons), bitmap, fill=True, color=1)

        if erosion>0:
            kernel = np.ones((5,5),np.uint8)
            bitmap = cv2.erode(bitmap,kernel,iterations = erosion)

        #get contours
        contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #get the min bounding rect for the rects
        min_area_conts = [np.int0(cv2.cv.BoxPoints(cv2.minAreaRect(cnt))) for cnt in contours]
        return min_area_conts


    def find_viola_proposals(self, viola_detections, img_name=None):
        '''Call viola to find coordinates of candidate roofs. 
        Extract those patches from the image, tranform them so they can be fed to neural network.
        Return both the coordinates and the patches.
        '''
        try:
            img_full = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            img_shape = img_full.shape
        except IOError as e:
            print e
            sys.exit(-1)

        all_proposal_patches = dict()
        all_proposal_coords = dict()
        
        #extract patches for neural network classification
        for roof_type in ['metal', 'thatch']: 
            all_proposal_coords[roof_type] = viola_detections.get_detections(img_name=img_name, roof_type=roof_type)
            #all_proposal_coords[roof_type] = self.viola.viola_detections.get_detections(img_name=img_name, roof_type=roof_type)
            patches = np.empty((len(all_proposal_coords[roof_type]), 3, utils.PATCH_W, utils.PATCH_H)) 

            for i, detection in enumerate(all_proposal_coords[roof_type]): 
                #extract the patch from the image using utils code
                img = utils.four_point_transform(img_full, detection)

                #transform the patch using utils code
                patch = utils.cv2_to_neural(img)
                patches[i, :, :,:] = patch 

            all_proposal_patches[roof_type] = patches  

        return all_proposal_patches, all_proposal_coords, img_shape


    def find_slider_proposals(self, slider_rects, img_name=None):
        #rects are in the form of (x, y, w, h)
        try:
            img_full = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            img_shape = img_full.shape
        except IOError as e:
            print e
            sys.exit(-1)

        all_proposal_patches = dict()
        
        #extract patches for neural network classification
        for roof_type in ['metal', 'thatch']: 
            patches = np.empty((len(slider_rects[roof_type]), 3, utils.PATCH_W, utils.PATCH_H)) 

            for i, rect in enumerate(slider_rects[roof_type]): 
                #extract the patch from the image using utils code
                img = img_full[rect.ymin:rect.ymax, rect.xmin:rect.xmax, :]
                #transform the patch using utils code
                patch = utils.cv2_to_neural(img)
                patches[i, :, :,:] = patch 

            all_proposal_patches[roof_type] = patches  

        return all_proposal_patches, img_shape



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
                    classes = np.array(self.net.test(proposal_patches[roof_type]))

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



    def neural_classification_AUC(self, proposal_patches, proposal_coords):
        #get the classification by evaluating it compared to the real roofs
        #get the probability of it being that type of roof
        classified_detections = dict()
        probs = dict()
        for roof_type in utils.ROOF_TYPES:
            classified_detections[roof_type] = list()
            if proposal_patches[roof_type].shape[0] > 1:
                probs[roof_type] = self.ensemble.predict_proba(proposal_patches[roof_type], roof_type=roof_type)
                #different detections depending on threshold
                coords = np.array(proposal_coords[roof_type])
                for thres in self.auc_thresholds:
                    detections_logical = probs[roof_type]>=thres   
                    classified_detections[roof_type].append(coords[detections_logical]) 
            else:
                print 'No {0} detections'.format(roof_type)
                raise ValueError('Need to fix this to support this case')
        return classified_detections, probs 
 

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


def setup_params(parameters, pipe_fname, method=None, decision='decideMean'):
    '''
    Get parameters for the pipeline and the components of the pipeline:
    the neural network(s) and the viola detector(s)
    '''
    preloaded_paths = parameters['preloaded_path'].split() #there can be more than one network, separated by space
    single_detector_boolean = False if len(preloaded_paths) == 2 else True

    #PIPE PARAMS: the grouping for each roof class parameter
    pipe_params = dict() 
    if 'metal_groupThres' in parameters:
        pipe_params['metal_groupThres'] = parameters['metal_groupThres']
    if 'thatch_groupThres' in parameters:
        pipe_params['thatch_groupThres'] = parameters['thatch_groupThres']

    neural_ensemble = Ensemble(preloaded_paths, scoring_strategy=decision, method=method)
    assert neural_ensemble is not None

    if method=='viola':
        #VIOLA PARAMS
        viola_params = dict()
        viola_data = neural_ensemble.data_path
        #a little messy, but we no longer call viola_data that, we call it data_path

        combo_fname = 'combo{0}'.format(int(utils.get_param_value_from_filename(viola_data,'combo')))
        viola_params['detector_names'] = viola_detector_helpers.get_detectors(combo_fname)

        #get other non-default params
        possible_parameters = ['min_neighbors','scale', 'group', 'removeOff', 'rotate', 'mergeFalsePos'] 
        for param in possible_parameters:
            if param in viola_data:
                viola_params[param]= utils.get_param_value_from_filename(viola_data, param)
        detector_params = viola_params

    elif method=='slide': #sliding window
        detector_params = dict()
        '''WE KNOW WHAT THESE PARAMS ARE, SO USE THEM DIRECTLY
        for param in ['scale', 'minSize', 'windowSize', 'stepSize']:
            if param == 'scale':
                detector_params[param] = float(parameters[param])
            elif param == 'stepSize':
                detector_params[param] = int(float(parameters[param]))
            else:
                detector_params[param] = (int(float(parameters[param])),int(float(parameters[param])))
        '''
        detector_params['scale'] = 1.3
        detector_params['minSize'] = (50,50)
        detector_params['windowSize'] = (15,15)
        detector_params['stepSize'] = 4

    else:
        print 'Unknown method of detection {}'.format(method)
        sys.exit(-1)
    return neural_ensemble, detector_params, pipe_params, single_detector_boolean


def get_main_param_filenum():
    #get the pipeline number
    viola_num = -1
    sliding_num = -1
    groupThres = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "v:s:d:g:")
    except getopt.GetoptError:
        print 'Command line error'
        sys.exit(2)  
    for opt, arg in opts:
        if opt == '-v':
            viola_num = int(float(arg))
        elif opt == '-s':
            sliding_num = int(float(arg))
        elif opt == '-g':
            groupThres = float(arg)
    return viola_num, sliding_num, groupThres 


if __name__ == '__main__':
    viola_num, sliding_num, groupThres = get_main_param_filenum()
    decision = 'decideMean'

    pickle_viola = False 
    groupBounds = False
    erosion = 0  

    #get the parameters from the pipeline
    if viola_num > 0:
        pipe_fname = 'viola{}.csv'.format(viola_num)
        method = 'viola'
    elif sliding_num > 0:
        pipe_fname = 'slide{}.csv'.format(sliding_num)
        method = 'slide'


    full_dataset = False 
    if full_dataset:
        print 'WARNING: USING FULL DATASET !!!!!!!!!!!!!!!!!!! '
    if method=='viola' and pickle_viola: 
        pickle_viola = None 
    else:
        pickle_viola = None

    print pipe_fname
    print method
    parameters = utils.get_params_from_file( '{0}{1}'.format(utils.get_path(params=True, pipe=True), pipe_fname))
    neural_ensemble, detector_params, pipe_params, single_detector_bool = setup_params(parameters, pipe_fname[:-len('.csv')], method=method, decision=decision)

    #if the groupthres is defined in the pipeline paramter file, we take it
    if 'thatch_groupThres' not in pipe_params:
        thatch_groupThres = groupThres
    else:
        thatch_groupThres = pipe_params['thatch_groupThres']
    if 'metal_groupThres' not in pipe_params:
        metal_groupThres = groupThres
    else:
        metal_groupThres = pipe_params['metal_groupThres']

    
    #UNCOMMENT THIS TO TEST NEW NETS WITH MULTTIPLE GROUPINGS
    #group = [.1,.2,.3,.4,.5] if groupThres is None else [groupThres]
    #for groupThres in group:
    #I/O
    out_folder_name = '{0}_metalGroup{1}_thatchGroup{2}'.format(pipe_fname[:-len('.csv')], metal_groupThres, thatch_groupThres) 
    in_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.IN, pipe=True, full_dataset=full_dataset) 
    out_path = utils.get_path(data_fold=utils.VALIDATION, in_or_out = utils.OUT, 
                                pipe=True, out_folder_name=out_folder_name, full_dataset=full_dataset)   
    pickle_auc = False 
    pipe = Pipeline(method=method, 
                    full_dataset=full_dataset,
                    pickle_viola=pickle_viola,# single_detector=single_detector_bool, 
                    in_path=in_path, out_path=out_path, 
                    groupBounds=groupBounds, 
                    ensemble=neural_ensemble, 
                    detector_params=detector_params, out_folder_name=pipe_fname, **pipe_params)  
    if pickle_auc == False:
        pipe.run()
        with open(out_path+'auc.pickle', 'wb') as f:
            pickle.dump(pipe.auc, f)
        pipe.auc.plot_auc()
    else:
        with open(out_path+'auc.pickle', 'rb') as f:
            auc = pickle.load(f)
        auc.plot_auc()


