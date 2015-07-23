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

DEBUG = True

class ViolaDetector(object):
    def __init__(self, 
            in_path=None, 
            out_path=None,
            folder_name=None,
            save_imgs=False,
            detector_names=None, 
            group=None, #minNeighbors, scale...
            min_neighbors=3,
            scale=1.1,
            rotate=False 
            ):
        assert in_path is not None
        self.in_path = in_path
        assert out_path is not None
        self.out_folder = out_path
        print 'Will output evaluation to: {0}'.format(self.out_folder)

        self.img_names = [f for f in os.listdir(in_path) if f.endswith('.jpg')]
        self.save_imgs = save_imgs

        #parameters for detection 
        self.scale = scale
        self.min_neighbors = min_neighbors
        self.group = group
        self.angles = utils.VIOLA_ANGLES if rotate else [0]

        self.viola_detections = Detections()
        self.setup_detectors(detector_names)

        self.evaluation = Evaluation(method='viola', folder_name=folder_name, out_path=self.out_folder, detections=self.viola_detections, 
                    in_path=self.in_path, detector_names=detector_names)


    def setup_detectors(self, detector_names=None, old_detector=False):
        '''Given a list of detector names, get the detectors specified
        '''
        #get the detectors
        assert detector_names is not None 
        self.roof_detectors = defaultdict(list)
        self.detector_names = detector_names

        #if old_detector:
        #    self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_names['metal']]
        #    self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_names['thatch']]

        #else:
        for roof_type in ['metal', 'thatch']:
            for path in detector_names[roof_type]: 
                if path.startswith('cascade'):
                    start = '../viola_jones/rectified/' if '_rect_' in path else '../viola_jones/'
                    self.roof_detectors[roof_type].append(cv2.CascadeClassifier(start+path+'/cascade.xml'))
                else:
                    self.roof_detectors[roof_type].append(cv2.CascadeClassifier('../viola_jones/cascade_'+path+'/cascade.xml'))


    def detect_roofs_in_img_folder(self):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        for i, img_name in enumerate(self.img_names):
            print 'Processing image {0}/{1}\t{2}'.format(i, len(self.img_names), img_name)
            _, _, rotated_images = self.detect_roofs(img_name)
            self.evaluation.score_img_rectified(img_name, rotated_images)
        self.evaluation.print_report()
        open(self.out_folder+'DONE', 'w').close() 


#, img_name=None, group=None, reject_levels=1.3, level_weights=5, scale=None, min_neighbors=1, eps=2
    def detect_roofs(self, img_name):
        #rotated_images = dict()
        try:
            img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        except IOError as e:
            print e
            sys.exit(-1)
        else:
            self.total_detection_time = 0
            detected_roofs = defaultdict(list)

            for roof_type in self.roof_detectors.keys():
                detected_roofs[roof_type] = defaultdict(list)
                for i, detector in enumerate(self.roof_detectors[roof_type]):

                    #ANGLE ROTATIONS AND DETECTIONS
                    for angle in self.angles:
                        #separate detections by angle

                        print 'Detecting with detector: '+str(i)
                        print 'ANGLE '+str(angle)

                        with Timer() as t: 
                            if angle == 0:
                                temp_detections = detector.detectMultiScale(gray, scaleFactor=self.scale, minNeighbors=self.min_neighbors )
                                final_detections = temp_detections
                                rotated_image = gray 
                                
                            else:
                                rotated_img = utils.rotate_image(gray, angle)
                                temp_detections =  detector.detectMultiScale(rotated_img, scaleFactor=self.scale, minNeighbors=self.min_neighbors)
                                final_detections = self.transform_points(temp_detections, rotated_img, angle)

                            detected_roofs[roof_type][angle].extend(final_detections)
                        print 'Time detection: {0}'.format(t.secs)
                        self.viola_detections.total_time += t.secs
                        print 'DETECTED {0} roofs'.format(len(detected_roofs[roof_type][angle]))
                        self.viola_detections.set_detections(img_name=img_name, angle=angle, detection_list=detected_roofs, img=rotated_img)

                        if DEBUG:
                            #you need to use the temp_detections because they contain the right coordinates on the rotated image
                            rgb_rotated = utils.rotate_image_RGB(img, angle)
                            self.mark_save_current_rotation(img_name, rgb_rotated, temp_detections, angle)

                        #GROUPING
                        # TODO: fix the grouping
                        # with Timer() as t:
                        #     if self.group == 'group_rectangles':
                        #         raise ValueError('We need to consider the angles here also')
                        #         detected_roofs[roof_type][angle], weights = cv2.groupRectangles(np.array(detected_roofs[roof_type]).tolist(), min_neighbors, eps)
                        #     elif self.group ==  'group_bounding':
                        #         raise ValueError('We need to consider the angles here also')
                        #         detected_roofs[roof_type][angle] = self.evaluation.get_bounding_rects(img_name=img_name, rows=1200, 
                        #                     cols=2000, detections=detected_roofs[roof_type]) 
                        #     else:
                        #         pass

                self.viola_detections.total_time += t.secs

            return detected_roofs, img #, rotated_images


    def mark_save_current_rotation(self, img_name, img, detections, angle):
        img = self.evaluation.mark_roofs_on_img(img_name=img_name, img=img, roofs=detections, color=(0,0,255))
        cv2.imwrite('{0}_angle{1}.jpg'.format(self.out_folder+img_name[:-4], angle), img)
        


    def transform_points(self, detections, img, theta):
        print 'transforming points...'
        final_detections = list()
        for x,y,w,h in detections:
            y, x = (utils.rotate_point((y,x), img, theta)) 
            final_detections.append((x,y,w,h))
        return final_detections




