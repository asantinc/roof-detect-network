import os
import subprocess
import pdb
import math

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import get_data
import experiment_settings as settings
from viola_trainer import ViolaTrainer, ViolaDataSetup


class ViolaDetector(object):
    def __init__(self, num_pos=None, num_neg=None, 
            detector_paths=None, 
            output_folder=None,
            save_imgs=False,
            scale=1.05
            ):
        self.num_pos = num_pos if num_pos is not None else 0
        self.num_neg = num_neg if num_neg is not None else 0
        self.scale = scale
        self.save_imgs = save_imgs

        #used to compare roof detections to ground truth
        self.total_roofs = 0
        self.true_positives = 0
        self.all_detections_made = 0  
        self.roofs_detected = dict()
        self.overlap_dict = dict()

        #get the detectors
        assert detector_paths is not None
        self.roof_detectors = list()
        self.detector_paths = detector_paths
        for path in self.detector_paths:
            self.roof_detectors.append(cv2.CascadeClassifier(path))

        #file name beginning
        self.file_name_default = ''
        if num_pos != 0 and num_neg != 0:
            self.file_name_default = '''numPos{0}_numNeg{1}_scale{2}'''.format(num_pos, num_neg, self.scale)
        
        #open report file
        assert output_folder is not None
        self.output_folder = output_folder
        self.report_file = self.output_folder+'report.txt'
        try:
            report = open(self.report_file, 'w')
            report.write('****************** DETECTORS USED ******************\n')
            report.write('\n'.join(detector_paths))
            report.write('\n')
        except IOError as e:
            print e
        else:
            report.close()

    def detect_roofs(self, img_name=None, img_path=None, reject_levels=1.3, level_weights=5, scale=None):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if scale is not None:
            self.scale = scale

        #get detections in the image
        roof_detections = list()
        for i, detector in enumerate(self.roof_detectors):
            print 'Detecting with detector: '+str(i)
            detected_roofs = detector.detectMultiScale(
                        gray,
                        scaleFactor=self.scale,
                        minNeighbors=5)
            #group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
            roof_detections.append(detected_roofs)
        self.roofs_detected[img_name] = roof_detections
        return self.roofs_detected[img_name]


    def compare_detections_to_roofs_folder(path=settings.INHABITED_PATH, reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        loader = get_data.DataLoader()
        img_names = get_data.DataLoader.get_img_names_from_path(path=path)
        for i, img_name in enumerate(img_names):
            self.compare_detections_to_roofs(img_name=img_name, path=path, reject_levels=reject_levels, level_weights=level_weights, scale=scale)


    def compare_detections_to_roofs(self, img_name='', path='', reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs in an image
        '''
        self.scale = scale
        img_path = path+img_name
        xml_path = path+img_name[:-3]+'xml'

        self.detect_roofs(img_name=img_name, img_path=img_path)
        self.report_roofs_detected(i, img_name)

        try:
            image = misc.imread(img_path)
        except IOError:
            print 'Cannot open '+img_path
        rows, cols, _ = image.shape
        roof_list, _, _ = loader.get_roofs(xml_path)
        
        self.match_roofs_to_detection(img_name, roof_list, rows, cols)
        self.print_report(img_name)

        if self.save_imgs:
            self.save_detection_img(img_path, img_name, roof_list)

        self.print_report(final_stats=True)


    def get_patch_mask(self, img_name, rows=1200, cols=2000):
        patch_mask = np.zeros((rows, cols), dtype=bool)
        patch_area = 0
        for i, cascade in enumerate(self.roofs_detected[img_name]):
            for (x,y,w,h) in cascade:
                patch_mask[y:y+h, x:x+w] = True
                patch_area += w*h
        return patch_mask, patch_area


    def get_detection_contours(self, patch_path, img_name):
        im_gray = cv2.imread(patch_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #find and save contours of detections
        output_contours = np.zeros((im_bw.shape[0],im_bw.shape[1]))
        cv2.drawContours(output_contours, contours, -1, (255,255,255), 3)
        bounding_path = self.output_folder+img_name+'_contour.jpg'
        cv2.imwrite(bounding_path, output_contours)

        #find and save bouding rectangles of contours
        output_bounds = np.zeros((im_bw.shape[0],im_bw.shape[1]))
        for cont in contours:
            bounding_rect = cv2.boundingRect(cont)
            cv2.rectangle(output_bounds, (bounding_rect[0],bounding_rect[1]),(bounding_rect[0]+bounding_rect[2],bounding_rect[1]+bounding_rect[3]), (255,255,255),5)
        #write to file
        bounding_path = self.output_folder+img_name+'_bound.jpg'
        cv2.imwrite(bounding_path, output_bounds)
        return contours

        
    def match_roofs_to_detection(self, img_name, roof_list, rows=1200, cols=2000):
        #Compare detections to ground truth
        self.overlap_dict[img_name] = list()
        patch_mask, patch_area = self.get_patch_mask(img_name)
        patch_location = self.output_folder+img_name+'_mask.jpg'
        #cv2.imwrite(patch_location, patch_mask.astype(np.int8))
        misc.imsave(patch_location, patch_mask)

        detection_contours = self.get_detection_contours(patch_location, img_name)
        for roof in roof_list:
            if roof.roof_type == self.roof_type:
                #self.overlap_dict[img_name].append(roof.max_overlap_single_patch(rows=rows, cols=cols,detections=self.roofs_detected[img_name]))
                self.overlap_dict[img_name].append(roof.check_overlap_total(patch_mask, patch_area, rows, cols))


    def report_roofs_detected(self, i, img_name):
        detected_numbers = list()
        for detection_set in self.roofs_detected[img_name]:
            detected_numbers.append(len(detection_set))
        print '*************************** IMAGE:'+str(i)+','+str(img_name)+'***************************'
        print 'Roofs detected: '+str(detected_numbers)+' Total: '+str(sum(detected_numbers))


    def print_report(self, img_name='', detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as output:
            if not final_stats:
                curr_detections_made = sum([len(detections) for detections in self.roofs_detected[img_name]])
                self.all_detections_made += curr_detections_made 

                current_img_roofs = len(self.overlap_dict[img_name])
                self.total_roofs += current_img_roofs
                current_img_true_positives = 0

                for v in self.overlap_dict[img_name]:
                    if v > 0.20:
                        current_img_true_positives += 1
                self.true_positives += current_img_true_positives
                log = 'Image '+img_name+': '+str(current_img_true_positives)+'/'+str(current_img_roofs)+'\n'
                print log
                output.write(log)

            elif final_stats:
                #Print number of false positives, False negatives
                pdb.set_trace()
                log = ('******************************* RESULTS ********************************* \n'
                    +'Precision: \t'+str(self.true_positives)+'/'+str(self.all_detections_made)+
                    '\n'+'Recall: \t'+str(self.true_positives)+'/'+str(self.total_roofs)+'\n')
                print log
                output.write(log)

                self.true_positive = 0
                self.total_roofs = 0
                self.all_detections_made = 0


    def mark_detections_on_img(self, img, img_name):
        ''' Save an image with the detections and the ground truth roofs marked with rectangles
        '''
        for i, cascade in enumerate(self.roofs_detected[img_name]):
            for (x,y,w,h) in cascade:
                if i%3==0:
                    color=(255,0,0)
                elif i%3==1:
                    color=(0,0,255)
                elif i%3==2:
                    color=(255,255,255)
                cv2.rectangle(img,(x,y),(x+w,y+h), color, 2)
        return img


    def mark_roofs_on_img(self, img, img_name, roof_list):
        for roof in roof_list:
            if roof.roof_type == self.roof_type:
                cv2.rectangle(img,(roof.xmin,roof.ymin),(roof.xmin+roof.width,roof.ymin+roof.height),(0,255,0),2)
        return img


    def save_detection_img(self, img_path, img_name, roof_list):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.mark_detections_on_img(img, img_name)
        img = self.mark_roofs_on_img(img, img_name, roof_list)
        self.save_image(img, img_name)


    def save_image(self, img, img_name, output_folder=None):
        if output_folder is not None:
            self.output_folder = output_folder 
        cv2.imwrite(self.output_folder+self.file_name_default+'_'+img_name, img)


if __name__ == '__main__':
    detectors = ['../viola_jones/cascade_thatch_5_augment/cascade.xml']
    output = '../output/viola_thatch_5_augment/'
    num_pos = 0
    num_neg = 0
    viola = ViolaDetector(num_pos, num_neg, detector_paths=detectors, output_folder=output, save_imgs=True)
    viola.compare_detections_to_roofs_folder(reject_levels=0.5, level_weights=2, scale=1.05)










