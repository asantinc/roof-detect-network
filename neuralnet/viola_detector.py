import os
import subprocess
import pdb
import math
from collections import defaultdict
import pickle
import csv

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import get_data
from get_data import Roof
import experiment_settings as settings
from timer import Timer

from reporting import Evaluation, Detections

            
class ViolaDetector(object):
    def __init__(self, 
            in_path=None, 
            detector_names=None, 
            out_path=settings.VIOLA_OUT,
            out_folder_name=None,
            save_imgs=False,
            scale=1.05,
            old_detector=True,
            ):
        assert in_path is not None
        self.img_names = [f for f in os.listdir(in_path) if f.endswith('.jpg')]
        self.scale = scale
        self.save_imgs = save_imgs

        self.viola_detections = Detections()
        self.setup_detectors(detector_names)

        self.in_path = in_path

        assert out_folder_name is not None
        self.out_folder = out_path+out_folder_name
        if not os.path.isdir(self.out_folder):
            subprocess.check_call('mkdir {0}'.format(self.out_folder), shell=True)
        print 'Will output evaluation to: {0}'.format(self.out_folder)

        self.evaluation = Evaluation(out_folder=self.out_folder, detections=self.viola_detections, 
                    in_path=self.in_path, img_names=self.img_names, detector_names=detector_names)


    def setup_detectors(self, detector_names=None, old_detector=False):
        '''Given a list of detector names, get the detectors specified
        '''
        #get the detectors
        assert detector_names is not None 
        self.roof_detectors = dict()
        self.detector_names = detector_names

        if old_detector:
            self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_names['metal']]
            self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_names['thatch']]

        else:
            self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/cascade_'+path+'/cascade.xml') for path in detector_names['metal']]
            self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/cascade_'+path+'/cascade.xml') for path in detector_names['thatch']]


    def detect_roofs_in_img_folder(self, reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        for i, img_name in enumerate(self.img_names):
            print 'Processing image {0}/{1}\t{2}'.format(i, len(self.img_names), img_name)
            self.detect_roofs(img_name=img_name, reject_levels=reject_levels, level_weights=level_weights, scale=scale)
            self.evaluation.score_img(img_name)
        self.evaluation.print_report()
        open(self.out_folder+'DONE', 'w').close() 


    def detect_roofs(self, img_name=None, reject_levels=1.3, level_weights=5, scale=None):
        img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        self.scale = scale if scale is not None else self.scale

        #get detections in the image
        self.total_detection_time = 0

        for roof_type in self.roof_detectors.keys():
            for i, detector in enumerate(self.roof_detectors[roof_type]):
                print 'Detecting with detector: '+str(i)
                with Timer() as t: 
                    detected_roofs = detector.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
                print 'Time: {0}'.format(t.secs)
                self.viola_detections.total_time += t.secs
                #group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
                self.viola_detections.set_detections(roof_type, img_name, detected_roofs)
                print 'DETECTED {0} roofs'.format(len(detected_roofs))




class NeuralViolaDetector(ViolaDetector):
    def __init__(self):
        super(NeuralViolaDetector, self).__init__()


    def get_neural_training_data(self, save_false_pos=False, roof_type=None, out_path=settings.TRAINING_NEURAL_PATH, detector_name=None):
        '''
        Run detection to get false positives, true positives from viola jones. Save these patches.
        These results will be used to then train a neural network.
        '''
        in_path = settings.TRAINING_PATH #since this is to train neural network, we can only use training data
        true_pos_list = list()
        false_pos_list = list()

        loader = get_data.DataLoader()
        img_names = get_data.DataLoader.get_img_names_from_path(path=in_path)

        for num, img_name in enumerate(img_names):
            print 'Processing image: '+str(num)+', '+img_name+'\n' 
            self.detect_roofs(img_name=img_name, img_path=in_path+img_name)
            
            #get roofs
            xml_path = in_path+img_name[:-3]+'xml'
            roof_list = loader.get_roofs(xml_path, img_name)

            #get true and false positives
            cur_true_pos, cur_false_pos = self.find_true_false_positives(1200, 2000, img_name, roof_type, roof_list, threshold=settings.NEAR_MISS) 

            true_pos_list.extend(cur_true_pos)
            false_pos_list.extend(cur_false_pos)

        with open(out_path+'{1}_true_pos_from_viola_{0}.pickle'.format(detector_name, roof_type), 'wb') as true_pos_file:
            pickle.dump(true_pos_list, true_pos_file) 
        with open(out_path+'{1}_false_pos_from_viola_{0}.pickle'.format(detector_name, roof_type), 'wb') as false_pos_file:
            pickle.dump(false_pos_list, false_pos_file)


    def save_patches_to_folder(self, in_path=None, img_name=None, out_path=None, patches=None):
        '''Save false pos and true pos to out_path. These patches will be used to train the neural network
        '''
        try:
            for i, (x,y,w,h) in enumerate(patches):
                try:
                    img = cv2.imread(in_path+img_name)
                except IOError:
                    print 'Cannot open '+in_path+img_name
                else:
                    try:
                        patch = img[y:(y+h), x:(x+w)]
                        cv2.imwrite(out_path+'_'+img_name+'_'+str(i)+'.jpg', patch)
                    except (IndexError, IOError, KeyError, ValueError) as e:
                        print e
        except TypeError as e:
            pdb.set_trace()


    def find_true_false_positives(self, img_rows, img_cols, img_name, roof_type, roof_list, threshold=settings.PARTIALLY_CLASSIFIED):
        '''Divide detections between true positives and false positives depending on the percentage of a roof they cover
        '''
        detections = self.viola_detections.get_img_detections_specific_type(roof_type, img_name)
        print 'Detect num: '+str(len(detections))+'\n'
        true_pos = list()
        false_pos_logical = np.empty(len(detections), dtype=bool)

        other_roof_type = 'metal' if roof_type == 'thatch' else 'thatcht'
        other_roofs = [(r.xmin, r.ymin, r.width, r.height) for r in roof_list if r.roof_type==other_roof_type]
        other_roofs_type_mask, percent_covered = self.get_patch_mask(detections=other_roofs)
        other_roofs_type_sum = Roof.sum_mask(other_roofs_type_mask)  

        for d, (x,y,w,h) in enumerate(detections):                           #for each patch found
            for roof in roof_list: #check whether match exists with any roof                   
                if roof.roof_type != roof_type:
                    continue
                roof_mask = np.zeros((img_rows, img_cols))
                roof_area = roof.width*roof.height
                roof_mask[roof.ymin:roof.ymin+roof.height, roof.xmin:roof.xmin+roof.width] = 1   #the roof
                roof_mask[y:y+h, x:x+w] = 0        #detection
                curr_miss = Roof.sum_mask(roof_mask)

                #true positives
                percent_found = (roof_area-curr_miss)*(1.)/roof_area
                if percent_found > settings.PARTIALLY_CLASSIFIED:
                    true_pos.append((x,y,w,h))
                    false_pos_logical[d] = False
                else:
                    #check whether it overlaps with the other type of roof (if the sum is lower now)
                    copy_other_roofs_type_mask = np.copy(other_roofs_type_mask) 
                    copy_other_roofs_type_mask[y:y+h, x:x+w] = 0 
                    if (other_roofs_type_sum - Roof.sum_mask(copy_other_roofs_type_mask))<5:
                        #if our detection covers very little of any roof of the opposite class, we don't consider this a negative patch
                        false_pos_logical[d] = True

        det = np.array(detections)
        false_pos = det[false_pos_logical]
        print 'Roofs:{2},  True Pos: {0}, False pos: {1}'.format(len(true_pos), len(false_pos), len(roof_list))
        return true_pos, false_pos 




