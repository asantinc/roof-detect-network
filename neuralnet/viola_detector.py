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
import utils
from timer import Timer

from reporting import Evaluation, Detections

            
class ViolaDetector(object):
    def __init__(self, 
            in_path=None, 
            detector_names=None, 
            out_path=None,
            out_folder_name=None,
            save_imgs=False,
            scale=1.05,
            old_detector=True,
            ):
        assert out_folder_name is not None
        assert in_path is not None
        assert out_path is not None

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

        self.evaluation = Evaluation(method='viola', folder_name=out_folder_name, out_path=self.out_folder, detections=self.viola_detections, 
                    in_path=self.in_path, detector_names=detector_names)


    def setup_detectors(self, detector_names=None, old_detector=False):
        '''Given a list of detector names, get the detectors specified
        '''
        #get the detectors
        assert detector_names is not None 
        self.roof_detectors = defaultdict(list)
        self.detector_names = detector_names
        print detector_names
        pdb.set_trace()
        if old_detector:
            self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_names['metal']]
            self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_names['thatch']]

        else:
            for roof_type in ['metal', 'thatch']:
                for path in detector_names[roof_type]: 
                    if path.startswith('cascade'):
                        self.roof_detectors[roof_type].append(cv2.CascadeClassifier('../viola_jones/'+path+'/cascade.xml'))
                    else:
                        self.roof_detectors[roof_type].append(cv2.CascadeClassifier('../viola_jones/cascade_'+path+'/cascade.xml'))


    def detect_roofs_in_img_folder(self, group=None, reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        for i, img_name in enumerate(self.img_names):
            print 'Processing image {0}/{1}\t{2}'.format(i, len(self.img_names), img_name)
            self.detect_roofs(group=group, img_name=img_name, reject_levels=reject_levels, level_weights=level_weights, scale=scale)
            self.evaluation.score_img(img_name)
        self.evaluation.print_report()
        open(self.out_folder+'DONE', 'w').close() 


    def detect_roofs(self, img_name=None, group=None, reject_levels=1.3, level_weights=5, scale=None, min_neighbors=1, eps=2):

        img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        self.scale = scale if scale is not None else self.scale

        #get detections in the image
        self.total_detection_time = 0
        detected_roofs = defaultdict(list)
        for roof_type in self.roof_detectors.keys():
            for i, detector in enumerate(self.roof_detectors[roof_type]):
                print 'Detecting with detector: '+str(i)
                checking =  detector.empty()
                print roof_type, checking
                with Timer() as t: 
                    detected_roofs[roof_type].extend(detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3))
                print 'Time: {0}'.format(t.secs)
                self.viola_detections.total_time += t.secs
                #group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
                print 'DETECTED {0} roofs'.format(len(detected_roofs))
            if group == 'group_rectangles':
                detected_roofs[roof_type], weights = cv2.groupRectangles(np.array(detected_roofs[roof_type]).tolist(), min_neighbors, eps)
            elif group ==  'group_bounding':
                detected_roofs[roof_type] = self.evaluation.get_bounding_rects(img_name=img_name, rows=1200, 
                            cols=2000, detections=detected_roofs[roof_type]) 
            else:
                pass
        self.viola_detections.set_detections(img_name=img_name, detection_list=detected_roofs)




