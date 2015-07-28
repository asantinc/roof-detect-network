import os
import sys
import getopt
import subprocess
import pdb
import math
from collections import defaultdict
import pickle
import csv
import itertools

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import utils
from timer import Timer

from reporting import Evaluation, Detections
import viola_detector_helpers
import suppression

DEBUG = False

class ViolaDetector(object):
    def __init__(self, 
            in_path=None, 
            out_path=None,
            folder_name=None,
            save_imgs=False,
            detector_names=None, 
            group=None, #minNeighbors, scale...
            overlapThresh=None,
            downsized=False,
            min_neighbors=3,
            scale=1.1,
            rotate=True, 
            removeOff=True,
            output_patches=True
            ):
        assert in_path is not None
        self.in_path = in_path
        assert out_path is not None
        self.out_folder = out_path
        self.out_folder_name = folder_name
        print 'Will output evaluation to: {0}'.format(self.out_folder)

        self.img_names = [f for f in os.listdir(in_path) if f.endswith('.jpg')]
        self.save_imgs = save_imgs
        self.output_patches = output_patches

        #parameters for detection 
        self.scale = scale
        self.min_neighbors = min_neighbors
        self.group = group
        self.overlapThresh = overlapThresh
        self.angles = utils.VIOLA_ANGLES if rotate else [0]
        self.remove_off_img = removeOff

        self.viola_detections = Detections()
        self.setup_detectors(detector_names)

        self.evaluation = Evaluation(method='viola', folder_name=folder_name, out_path=self.out_folder, detections=self.viola_detections, 
                    in_path=self.in_path, detector_names=detector_names, output_patches=output_patches)


    def setup_detectors(self, detector_names=None, old_detector=False):
        '''Given a list of detector names, get the detectors specified
        '''
        #get the detectors
        assert detector_names is not None 
        self.roof_detectors = defaultdict(list)
        self.detector_names = detector_names

        for roof_type in utils.ROOF_TYPES:
            for path in detector_names[roof_type]: 
                if path.startswith('cascade'):
                    start = '../viola_jones/cascades/' 
                    self.roof_detectors[roof_type].append(cv2.CascadeClassifier(start+path+'/cascade.xml'))
                    assert self.roof_detectors[roof_type][-1].empty() == False
                else:
                    self.roof_detectors[roof_type].append(cv2.CascadeClassifier('../viola_jones/cascade_'+path+'/cascade.xml'))


    def detect_roofs_in_img_folder(self):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        for i, img_name in enumerate(self.img_names):
            print '************************ Processing image {0}/{1}\t{2} ************************'.format(i, len(self.img_names), img_name)
            if self.group:
                img = self.detect_roofs_group(img_name)
            else:
                img = self.detect_roofs(img_name)
            self.evaluation.score_img(img_name, img.shape)
        self.evaluation.print_report()
        if self.output_patches:
            self.save_training_FP_and_TP(viola=True)
            self.save_training_FP_and_TP(neural=True)
        open(self.out_folder+'DONE', 'w').close() 


    def detect_roofs(self, img_name):
        try:
            rgb_unrotated = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            rgb_unrotated = utils.resize_rgb(rgb_unrotated, h=rgb_unrotated.shape[0]/2, w=rgb_unrotated.shape[1]/2)
            pdb.set_trace()
            gray = cv2.cvtColor(rgb_unrotated, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = utils.resize_rgb(gray, w=gray.shape[1]/2, h=gray.shape[0]/2) 
        except IOError as e:
            print e
            sys.exit(-1)
        else:
            for roof_type, detectors in self.roof_detectors.iteritems():
                for i, detector in enumerate(detectors):
                    for angle in self.angles:
                        #for thatch we only need one angle
                        if roof_type == 'thatch' and angle>0:
                            continue

                        print 'Detecting with detector: '+str(i)
                        print 'ANGLE '+str(angle)

                        with Timer() as t: 
                            rotated_image = utils.rotate_image(gray, angle) if angle>0 else gray
                            detections, _ = self.detect_and_rectify(detector, rotated_image, angle, rgb_unrotated.shape[:2]) 
                        print 'Time detection: {0}'.format(t.secs)
                        self.viola_detections.total_time += t.secs
                        self.viola_detections.set_detections(roof_type=roof_type, img_name=img_name, 
                                angle=angle, detection_list=detections, img=rotated_image)

                        if DEBUG:
                            rgb_to_write = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
                            utils.draw_detections(detections, rgb_to_write)
                            cv2.imwrite('{0}{1}_{2}.jpg'.format(self.out_folder, img_name[:-4], angle), rgb_to_write)
            return rgb_unrotated


    def detect_and_rectify(self, detector, image, angle, dest_img_shape):
        #do the detection
        detections = detector.detectMultiScale(image, scaleFactor=self.scale, minNeighbors=self.min_neighbors)
        #convert to proper coordinate system
        polygons = utils.convert_detections_to_polygons(detections)

        if angle > 0:
            #rotate back to original image coordinates
            print 'rotating...'
            rectified_detections = utils.rotate_detection_polygons(polygons, image, angle, dest_img_shape, remove_off_img=self.remove_off_img)
        else:
            rectified_detections = polygons
        print 'done rotating'

        if self.group:
            bounding_boxes = utils.get_bounding_boxes(np.array(rectified_detections))
        else:
            bounding_boxes = None
        return rectified_detections, bounding_boxes


    def detect_roofs_group(self, img_name):
        try:
            rgb_unrotated = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(rgb_unrotated, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        except IOError as e:
            print e
            sys.exit(-1)
        else:
            for roof_type, detectors in self.roof_detectors.iteritems():
                all_detections = list()
                for i, detector in enumerate(detectors):
                    for angle in self.angles:
                        #for thatch we only need one angle
                        if roof_type == 'thatch' and angle>0:
                            continue

                        print 'Detecting with detector: '+str(i)
                        print 'ANGLE '+str(angle)

                        with Timer() as t: 
                            rotated_image = utils.rotate_image(gray, angle) if angle>0 else gray
                            detections, bounding_boxes = self.detect_and_rectify(detector, rotated_image, angle, rgb_unrotated.shape[:2])
                            all_detections.append(list(bounding_boxes)) 
                        print 'Time detection: {0}'.format(t.secs)
                        self.viola_detections.total_time += t.secs
                #grouping
                all_detections = [d for detections in all_detections for d in detections] 
                grouped_detections, rects_grouped = cv2.groupRectangles(all_detections, 1) 
                print "GROUPING DOWN BY:"
                print len(all_detections)-len(grouped_detections)
                grouped_polygons = utils.convert_detections_to_polygons(grouped_detections)

                #merge the detections from all angles
                self.viola_detections.set_detections(roof_type=roof_type, img_name=img_name, 
                                detection_list=grouped_polygons, img=rotated_image)
            return rgb_unrotated



    def save_training_FP_and_TP(self, viola=False, neural=False):
        '''Save the correct and incorrect detections so that the neural network can train on it
        '''
        #we want to write to the params folder of neuralnet
        assert viola or neural
        general_path = utils.get_path(neural=neural, viola=viola, data_fold=utils.TRAINING, in_or_out=utils.IN, out_folder_name=self.out_folder_name) 

        path_true = general_path+'truepos_from_viola_training/'
        utils.mkdir(path_true)
         
        path_false = general_path+'falsepos_from_viola_training/'
        utils.mkdir(path_false)

        for img_name in self.img_names:
            try:
                if viola: #viola training will need grayscale patches
                    img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.equalizeHist(img)
                else: #neural network will need RGB
                    img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
            except:
                print 'Cannot open image'
                sys.exit(-1)
            #for the bad/background detections there is only one type: background (we've merge the thatch and metal false positives)
            detections = self.viola_detections.bad_detections[img_name]
            patches_path= path_false
            extraction_type = 'bad'
            roof_type = 'background'
            self.save_training_FP_and_TP_helper(img_name, detections, patches_path, general_path, img, roof_type, extraction_type, (0,0,255)) 
            #for viola we only need to save the negatives, since the positives can't be used (they'd need to be annotated and rectified)
            if neural:
                #for the good detections, we separate them between thatch and metal true positives
                for roof_type in utils.ROOF_TYPES:
                    detections = self.viola_detections.good_detections[roof_type][img_name] 
                    patches_path = path_true
                    extraction_type = 'good' 
                    self.save_training_FP_and_TP_helper(img_name, detections, patches_path, general_path, img, roof_type, extraction_type, (0,255,0))               


    def save_training_FP_and_TP_helper(self, img_name, detections, patches_path, general_path, img, roof_type, extraction_type, color):
        #this is where we write the detections we're extraction. One image per roof type
        #we save: 1. the patches and 2. the image with marks of what the detections are, along with the true roofs (for debugging)
        img_debug = np.copy(img) 

        if roof_type == 'background':
            utils.draw_detections(self.evaluation.correct_roofs['metal'][img_name], img_debug, color=(0, 0, 0), thickness=2)
            utils.draw_detections(self.evaluation.correct_roofs['thatch'][img_name], img_debug, color=(0, 0, 0), thickness=2)
        else:
            utils.draw_detections(self.evaluation.correct_roofs[roof_type][img_name], img_debug, color=(0, 0, 0), thickness=2)

        for i, detection in enumerate(detections):
            #extract the patch, rotate it to a horizontal orientation, save it
            warped_patch = utils.four_point_transform(img, detection)
            cv2.imwrite('{0}{1}_{2}_roof{3}.jpg'.format(patches_path, roof_type, img_name[:-4], i), warped_patch)
            
            #mark where roofs where taken out from for debugging
            utils.draw_polygon(detection, img_debug, fill=False, color=color, thickness=2, number=i)

        #write this type of extraction and the roofs to an image
        cv2.imwrite('{0}{1}_{2}_extract_{3}.jpg'.format(general_path, img_name[:-4], roof_type, extraction_type), img_debug)


    def mark_save_current_rotation(self, img_name, img, detections, angle, out_folder=None):
        out_folder = self.out_folder if out_folder is None else out_folder
        polygons = np.zeros((len(detections), 4, 2))
        for i, d in enumerate(detections):
            polygons[i, :] = utils.convert_rect_to_polygon(d)
        img = self.evaluation.mark_roofs_on_img(img_name=img_name, img=img, roofs=polygons, color=(0,0,255))
        path = '{0}_angle{1}.jpg'.format(out_folder+img_name[:-4], angle)
        print path
        cv2.imwrite(path, img)
 



def main(output_patches=True, detector_params=None, original_dataset=True, save_imgs=True, data_fold=utils.VALIDATION):
    combo_f_name = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:")
    except getopt.GetoptError:
        sys.exit(2)
        print 'Command line failed'
    for opt, arg in opts:
        if opt == '-f':
            combo_f_name = arg

    assert combo_f_name is not None
    detector = viola_detector_helpers.get_detectors(combo_f_name)

    viola = False if data_fold == utils.TRAINING else True
    in_path = utils.get_path(viola=viola, in_or_out=utils.IN, data_fold=data_fold)

    #name the output_folder
    folder_name = ['combo'+combo_f_name]
    for k, v in detector_params.iteritems():
        folder_name.append('{0}{1}'.format(k,v))
    folder_name = '_'.join(folder_name)

    out_path = utils.get_path(out_folder_name=folder_name, viola=True, in_or_out=utils.OUT, data_fold=data_fold)
    viola = ViolaDetector(output_patches=output_patches, out_path=out_path, in_path=in_path, folder_name = folder_name, save_imgs=save_imgs, 
                                        detector_names=detector,  **detector_params)
    viola.detect_roofs_in_img_folder()



if __name__ == '__main__':
    output_patches = False#True #if you want to save the true pos and false pos detections, you need to use the training set
    if output_patches:
        data_fold=utils.TRAINING
    else: 
        data_fold=utils.VALIDATION
    
    # removeOff: whether to remove the roofs that fall off the image when rotating (especially the ones on the edge
    #group: can be None, group_rectangles, group_bounding
    # if check_both_detectors is True we check if either the metal or the thatch detector has found a detection that matches either type of roof 
    detector_params = {'min_neighbors':3, 'scale':1.08, 'group': False, 'downsized':True, 'rotate':True, 'removeOff':True} 
    main(output_patches=output_patches, detector_params=detector_params, save_imgs=False, data_fold=data_fold, original_dataset=True)
 
