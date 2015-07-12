import os
import subprocess
import pdb
import math
from collections import defaultdict
import pickle

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import get_data
import experiment_settings as settings
from viola_trainer import ViolaTrainer, ViolaDataSetup
from timer import Timer

class ViolaDetector(object):
    def __init__(self, params_f=None, 
            detector_paths=None, 
            output_folder=None,
            save_imgs=False,
            scale=1.05,
            old_detector=True
            ):
        self.scale = scale
        self.save_imgs = save_imgs

        #used to compare roof detections to ground truth
        self.total_roofs = 0
        self.tp_metal = 0
        self.tp_thatch = 0
        self.metal_candidates = self.thatch_candidates = 0
        self.roofs_detected = dict()
        self.overlap_dict = dict()

        #get the detectors
        assert detector_paths is not None 
        self.detector_paths = detector_paths
        self.roof_detectors = dict()

        if old_detector:
            self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_paths['metal']]
            self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/cascades/'+path) for path in detector_paths['thatch']]

        else:
            self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/'+path+'/cascade.xml') for path in detector_paths['metal']]
            self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/'+path+'/cascade.xml') for path in detector_paths['thatch']]

        #file name beginning
        metal_name = '+'.join(detector_paths['metal'])+'+' if len(detector_paths['metal'])>0 else ''
        metal_name += '+'.join(detector_paths['thatch']) if len(detector_paths['thatch'])>0 else metal_name[:-1]
        #self.out_file = metal_name
        self.out_file = params_f

        #make a folder if it doens't exist
        '''
        for det in self.detector_paths['metal']:
            mkdir_cmd = 'mkdir {0}{1}'.format(settings.VIOLA_OUT, det)
            try:
                subprocess.check_call(mkdir_cmd, shell=True)
            except Exception as e:
                print e
        for det in self.detector_paths['thatch']:
            mkdir_cmd = 'mkdir {0}'.format(det)
            try:
                subprocess.check_call(mkdir_cmd, shell=True)
            except Exception as e:
                print e
        '''

        #open report file and create output folder if it doesn't exist
        self.output_folder = settings.VIOLA_OUT+params_f
        if not os.path.isdir(self.output_folder):
            subprocess.check_call('mkdir {0}'.format(self.output_folder), shell=True)
        print 'Will output files to: {0}'.format(self.output_folder)
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
        if os.path.isfile(self.output_folder+img_name+'.pickle'):
            #load from file
            with open(self.output_folder+img_name+'.pickle', 'rb') as f:
                self.roofs_detected[img_name] = pickle.load(f) 
        else:
            img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.scale = scale if scale is not None else self.scale

            #get detections in the image
            roof_detections = dict()
            for roof_type in self.roof_detectors.keys():
                roof_detections[roof_type] = list()
                for i, detector in enumerate(self.roof_detectors[roof_type]):
                    print 'Detecting with detector: '+str(i)
                    with Timer() as t: 
                        detected_roofs = detector.detectMultiScale(gray, scaleFactor=self.scale, minNeighbors=5)
                    print '{0} took {1} secs with detector {2} '.format(img_name, t.secs, i)
                    #group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
                    roof_detections[roof_type].append(detected_roofs)

            self.roofs_detected[img_name] = roof_detections
           
            #save detections to file
            with open(self.output_folder+img_name+'.pickle', 'wb') as f:
                pickle.dump(self.roofs_detected[img_name], f)
                 

    def compare_detections_to_roofs_folder(self, path=settings.INHABITED_PATH, save_detections=False, reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        loader = get_data.DataLoader()
        img_names = get_data.DataLoader.get_img_names_from_path(path=path)
        for i, img_name in enumerate(img_names):
            self.compare_detections_to_roofs(loader, img_name=img_name, path=path, reject_levels=reject_levels, level_weights=level_weights, scale=scale)
        self.print_report(final_stats=True)
   

    def compare_detections_to_roofs(self, 
            loader, img_name='', path='', 
            reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs in an image
        '''
        self.scale = scale
        img_path = path+img_name
        xml_path = path+img_name[:-3]+'xml'
        
        self.detect_roofs(img_name=img_name, img_path=img_path) 
        self.report_roofs_detected(img_name)
        roof_list = loader.get_roofs(xml_path, img_path)
        self.save_detection_img(img_path, img_name, roof_list)

        try:
            image = cv2.imread(img_path)
        except IOError:
            print 'Cannot open '+img_path
        rows, cols, _ = image.shape
        
        self.match_roofs_to_detection(img_name, roof_list, rows, cols)
        self.print_report(img_name)



    def get_patch_mask(self, img_name, rows=1200, cols=2000):
        '''Mark the area of a detection as True, everything else as False
        '''
        patch_mask = np.zeros((rows, cols), dtype=bool)
        patch_area = 0
        for roof_type in self.roofs_detected[img_name].keys():
            for i, detection in enumerate(self.roofs_detected[img_name][roof_type]):
                for (x,y,w,h) in detection:
                    patch_mask[y:y+h, x:x+w] = True
                    patch_area += w*h
        return patch_mask, float(patch_area)/(rows*cols)


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
        self.overlap_dict[img_name] = defaultdict(list)
        patch_mask, detection_percent = self.get_patch_mask(img_name)
        print 'Percent of image covered by detection:{0}'.format(detection_percent)
        patch_location = self.output_folder+img_name+'_mask.jpg'
        
        cv2.imwrite(patch_location, np.array(patch_mask, dtype=int))

        detection_contours = self.get_detection_contours(patch_location, img_name)
        for roof in roof_list:
            self.overlap_dict[img_name][roof.roof_type].append(roof.check_overlap_total(patch_mask, rows, cols))
            roof.max_overlap_single_patch(detections=self.roofs_detected[img_name])


    def report_roofs_detected(self, img_name):
        for roof_type in ['metal', 'thatch']:
            for i, detector in enumerate(self.detector_paths[roof_type]):
                print 'Detector {0}: \t {1}'.format(detector, len(self.roofs_detected[img_name][roof_type][i]) )


    def print_report(self, img_name='', detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as report:
            if not final_stats:
                self.thatch_candidates += len(self.overlap_dict[img_name]['thatch'])
                self.metal_candidates += len(self.overlap_dict[img_name]['metal'])
                
                #report true positives
                true_metal = true_thatch = 0
                for roof_type in self.overlap_dict.keys():
                    print roof_type
              
                    print self.overlap_dict[img_name][roof_type]
                    for v in self.overlap_dict[img_name][roof_type]:
                        if v > .20:
                            if roof_type == 'metal':
                                true_metal += 1
                            elif roof_type == 'thatch':
                                true_thatch += 1
                            else:
                                raise ValueError('Roof type {0} does not exist.'.format(roof_type))
                self.tp_metal += true_metal
                self.tp_thatch += true_thatch
                log = 'Image '+img_name+' thatch: '+str(true_thatch)+'/'+str(len(self.overlap_dict[img_name]['thatch']))+'\n'
                log = 'Image '+img_name+' metal: '+str(true_metal)+'/'+str(len(self.overlap_dict[img_name]['metal']))+'\n'
                print log
                report.write(log)

            else:
                #Print number of false positives, False negatives
                log = ('******************************* RESULTS ********************************* \n'
                    +'METAL: \n'+
                    +'Precision: \t'+str(self.tp_metal)+'/'+str(self.metal_candidates)+
                    +'\n'+'Recall: \t'+str(self.tp_metal)+'/'+str(self.all_true_metal)+'\n'+
                    +'THATCH: \n'+
                    +'Precision: \t'+str(self.tp_thatch)+'/'+str(self.thatch_candidates)+
                    +'\n'+'Recall: \t'+str(self.tp_thatch)+'/'+str(self.all_true_thatch)+'\n')
                print log
                report.write(log)


    def mark_detections_on_img(self, img, img_name):
        ''' Save an image with the detections and the ground truth roofs marked with rectangles
        '''
        for f in range(len(self.roofs_detected[img_name])):
            for i, (x,y,w,h) in enumerate(self.roofs_detected[img_name]['metal'][f]):
                if f%3==0:
                    color=(255,0,0)
                elif f%3==1:
                    color=(0,0,255)
                elif f%3==2:
                    color=(255,255,255)
                cv2.rectangle(img,(x,y),(x+w,y+h), color, 2)
        return img


    def mark_roofs_on_img(self, img, img_name, roof_list):
        for roof in roof_list:
            cv2.rectangle(img,(roof.xmin,roof.ymin),(roof.xmin+roof.width,roof.ymin+roof.height),(0,255,0),2)
        return img


    def save_detection_img(self, img_path, img_name, roof_list):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.mark_detections_on_img(img, img_name)
        img = self.mark_roofs_on_img(img, img_name, roof_list)
        self.save_image(img, img_name)


    def save_image(self, img, img_name, output_folder=None):
        self.output_folder = output_folder if output_folder is not None else self.output_folder  
        cv2.imwrite(self.output_folder+'_'+img_name, img)



if __name__ == '__main__':
    #can have multiple detectors for each type of roof
    detectors = dict()
    #detectors['metal'] = [ 'cascade_metal_0_square_augment_num3088_w24_h24','cascade_metal_0_tall_augment_num1608_w12_h24','cascade_metal_0_wide_augment_num2184_w24_h12']
    detectors['metal'] = ['cascade_w25_h12_pos393_neg700.xml', 'cascade_w12_h25_pos393_neg700.xml']
    detectors['thatch'] = []

    output = '../output/viola/'
    params_f = 'metal_non_augment_neg700_pos393/'
    viola = ViolaDetector(params_f=params_f, detector_paths=detectors, output_folder=output, save_imgs=True, old_detector = True)
    viola.compare_detections_to_roofs_folder(save_detections=True, reject_levels=0.5, level_weights=2, scale=1.05)










