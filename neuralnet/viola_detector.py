import os
import subprocess
import pdb
import getopt
import sys
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


class Detections(object):
    def __init__(self, roof_type=None, name=None):
        self.detections = dict()
        self.total_detection_num = 0
        self.total_time = 0
        self.imgs = set()
        self.roof_types = ['metal', 'thatch']
        for roof_type in self.roof_types:
            self.detections[roof_type] = defaultdict(list)

    def set_detections(self, roof_type=None, img_name=None, detection_list = None):
        assert roof_type is not None
        assert img_name is not None
        assert detection_list is not None
        if img_name not in self.imgs:
            self.imgs.update(img_name)
        self.total_detection_num += len(detection_list)
        self.detections[roof_type][img_name].extend(detection_list)        

    def get_img_detections_specific_type(self, roof_type, img_name):
        return self.detections[roof_type][img_name] 

    def get_img_detections_any_type(self, img_name):
        return self.detections[self.roof_types[0]][img_name]+self.detections[self.roof_types[1]][img_name]        

#    def get_all_detections(self):
#        for roof_type in self.roof_types:
            


class ViolaDetector(object):
    def __init__(self, 
            folder_name=None, 
            detector_names=None, 
            out_path=settings.VIOLA_OUT,
            save_imgs=False,
            scale=1.05,
            old_detector=True,
            pickle_dump_stats = True,
            load_pickle_stats = False,
            neural=False
            ):
        self.neural = neural
        self.scale = scale
        self.save_imgs = save_imgs
        self.pickle_dump_stats = pickle_dump_stats
        self.load_pickle_stats = load_pickle_stats

        #used to compare roof detections to ground truth
        self.total_roofs = 0
        self.tp_metal = 0
        self.tp_thatch = 0
        self.metal_candidates = self.thatch_candidates = 0
        self.viola_detections = Detections()
        self.overlap_roof_with_detections = defaultdict(list)
        
        self.setup_detectors(detector_names)

        self.out_file = folder_name 
        self.overlap_roof_with_detections = defaultdict()

        #open report file and create output folder if it doesn't exist
        if not neural:
            self.output_folder = out_path+folder_name
            if not os.path.isdir(self.output_folder):
                subprocess.check_call('mkdir {0}'.format(self.output_folder), shell=True)
            print 'Will output files to: {0}'.format(self.output_folder)
            self.report_file = self.output_folder+'report.txt'
            try:
                report = open(self.report_file, 'w')
                report.write('metal detectors: \t')
                report.write('\t'.join(detector_names['metal']))
                
                report.write('thatch_detector: \t')
                report.write('\t'.join(detector_names['thatch']))           
                report.write('\n')
            except IOError as e:
                print e
            else:
                report.close()


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

        print self.roof_detectors



    def detect_roofs(self, img_name=None, img_path=None, reject_levels=1.3, level_weights=5, scale=None):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        self.scale = scale if scale is not None else self.scale

        #get detections in the image
        self.total_detection_time = 0
        rejectLevels= list()
        LevelWeights=list()
        for roof_type in self.roof_detectors.keys():
            for i, detector in enumerate(self.roof_detectors[roof_type]):
                print 'Detecting with detector: '+str(i)
                with Timer() as t: 
                    detected_roofs = detector.detectMultiScale(gray, scaleFactor=self.scale, minNeighbors=5)
                print 'Time: {0}'.format(t.secs)
                self.viola_detections.total_time += t.secs
                #group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
                self.viola_detections.set_detections(roof_type, img_name, detected_roofs)
                print 'DETECTED {0} roofs'.format(len(detected_roofs))
            

    def detect_evaluate_roofs_folder(self, path=settings.VALIDATION_PATH, save_detections=False, reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        if self.load_pickle_stats == False:
            loader = get_data.DataLoader()
            img_names = get_data.DataLoader.get_img_names_from_path(path=path)
            for i, img_name in enumerate(img_names):
                print 'Processing image {0}/{1}\t{2}'.format(i, len(img_names), img_name)
                self.detect_evaluate_roofs(loader, img_name=img_name, path=path, reject_levels=reject_levels, level_weights=level_weights, scale=scale)
             
            #we can store the stats to use them directly
            if self.pickle_dump_stats:
                with open(self.output_folder+'stats.pickle', 'wb') as f:
                    pickle.dump(self.overlap_roof_with_detections, f)
        else:
            with open(self.output_folder+'stats.pickle', 'rb') as f:
                self.overlap_roof_with_detections = pickle.load(f)

        self.print_report()
        open(self.output_folder+'DONE', 'w').close() 

    def detect_evaluate_roofs(self, 
            loader, img_name='', path='', 
            reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs in an image
        '''
        self.scale = scale
        img_path = path+img_name
        xml_path = path+img_name[:-3]+'xml'
        self.detect_roofs(img_name=img_name, img_path=img_path) 
        #self.report_roofs_detected(img_name)
        roof_list = loader.get_roofs(xml_path, img_path)
        self.save_detection_img(img_path, img_name, roof_list)

        try:
            image = cv2.imread(img_path)
        except IOError:
            print 'Cannot open '+img_path
        rows, cols, _ = image.shape
        self.match_roofs_to_detection(img_name, roof_list, rows, cols)



    def get_patch_mask(self, img_name=None, detections=None, rows=1200, cols=2000):
        '''Mark the area of a detection as True, everything else as False
        '''
        if detections == None:#we either have a list of detections or an img_name
            assert img_name is not None
        detections = self.viola_detections.get_img_detections_any_type(img_name) if detections is None else detections
        patch_mask = np.zeros((rows, cols), dtype=bool)
        patch_area = 0
        for (x,y,w,h) in detections:
            patch_mask[y:y+h, x:x+w] = True
            patch_area += w*h

        area_covered = np.sum(patch_area)
        return patch_mask, float(area_covered)/(rows*cols) 


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

            if save_false_pos:
                self.mark_false_true_detections_and_roofs(out_path, in_path, img_name, roof_list, 
                                                           cur_false_pos,  cur_true_pos)

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


    def print_report(self):
        '''
        Write to file:
        - the number of roofs detected per file.
        - the false positives and the false negative rates of the joint detector
        - the average time taken for the detection
        '''
        fully_classified = defaultdict()
        mostly_classified = defaultdict()
        partially_classified = defaultdict()
        missed = defaultdict(int)
        near_miss = defaultdict(int)

        #single metric
        total_area_percentage_covered = defaultdict()
        total_area = defaultdict(int)
        total_possible_area = defaultdict(int)
        for roof_type in  ['thatch', 'metal']:
            total_area[roof_type] = defaultdict(int)
            total_possible_area[roof_type]=defaultdict(int)
            total_area_percentage_covered[roof_type] = defaultdict(int)
#            for cov in ['any', 'single']:
#                total_area[roof_type][cov] = defaultdict(int)
#                total_possible_area[roof_type][cov] = defaultdict(int)     

        for roof_type in ['thatch', 'metal']:
            fully_classified[roof_type] = defaultdict(int)
            mostly_classified[roof_type] = defaultdict(int)
            partially_classified[roof_type] = defaultdict(int)
            missed[roof_type] = defaultdict(int)
            near_miss[roof_type] =  defaultdict(int)

            #for cov in ['any', 'single']:
                #fully_classified[roof_type][cov] = defaultdict(int)
                #mostly_classified[roof_type][cov] = defaultdict(int)
                #partially_classified[roof_type][cov] = defaultdict(int)
                #missed[roof_type][cov] = defaultdict(int)

            for img_name in self.overlap_roof_with_detections.keys():
                for coverage_type in ['any', 'single']:
                    score_list =  self.overlap_roof_with_detections[img_name][roof_type][coverage_type]

                    total_area[roof_type][coverage_type] += sum(score_list)
                    total_possible_area[roof_type][coverage_type] += len(self.overlap_roof_with_detections[img_name][roof_type][coverage_type])
                    for roof_score in score_list:
                        if roof_score >= settings.FULLY_CLASSIFIED:
                            fully_classified[roof_type][coverage_type] += 1
                        elif roof_score>= settings.MOSTLY_CLASSIFIED:
                            mostly_classified[roof_type][coverage_type] += 1
                        elif roof_score >= settings.PARTIALLY_CLASSIFIED:
                            partially_classified[roof_type][coverage_type] += 1
                        elif roof_score >= settings.NEAR_MISS:
                            near_miss[roof_type][coverage_type] += 1
                        else:
                            missed[roof_type][coverage_type] += 1
            
            for coverage_type in ['any', 'single']:
                total_area_percentage_covered[roof_type][coverage_type] = float(total_area[roof_type][coverage_type])/total_possible_area[roof_type][coverage_type]

        #dictionary containing stats that we can pickle and unpickle for comparisons between runs
        evaluation = dict() 
        evaluation['path'] = self.output_folder
        with open(self.report_file, 'a') as report:
            #Print total detections
            detection_num = self.viola_detections.total_detection_num
            evaluation['detection_num']=detection_num            
            report.write('Detections:\t\t{0}'.format(detection_num)+'\n')
            #Print timing
            evaluation['timing'] = self.viola_detections.total_time

            for roof_type in ['metal', 'thatch']:
                if self.detector_names[roof_type] == []:
                    evaluation[roof_type] = False
                    continue
                evaluation[roof_type] = defaultdict()#dict for different converate types
                report.write('{0}'.format(roof_type))
                report.write('\t\t>{0}\t\t>{1}\t\t>{2}\t\t>{3}\t\tMissed\n'.format(settings.FULLY_CLASSIFIED, settings.MOSTLY_CLASSIFIED, settings.PARTIALLY_CLASSIFIED, settings.NEAR_MISS))
                for coverage_type in ['any', 'single']:
                    evaluation[roof_type][coverage_type] = defaultdict(list)#dict for actual metrics (in lists)
                    full = fully_classified[roof_type][coverage_type]
                    partial = partially_classified[roof_type][coverage_type]
                    mostly = partially_classified[roof_type][coverage_type]
                    near = near_miss[roof_type][coverage_type]
                    miss = missed[roof_type][coverage_type]
                    total = full+mostly+partial+near+miss
                    if total == 0:
                        continue

                    report.write('{0}\n'.format(coverage_type))
                    report.write('{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}\t\t{5}\n'.format('Roofs', full, mostly, partial, near, miss))
                    evaluation[roof_type][coverage_type]['roofs'] = [full, mostly, partial, near, miss]
                           
                    #write RECALL and PRECISION with the number of detections            
                    #report.write('Recall:\t\t{0:.{s}f}\t\t{1:.{s}f}\t\t{2:.{s}f}\t\t{3:.{s}f}\t\t{4:.{s}f}\n'.format(float(full)/total,float(full+mostly)/total, float(full+mostly+partial)/total, 
                    #    float(full+mostly+partial+near)/total, float(full+mostly+partial+near+miss)/total))
                    float_list = [float(full)/total,float(full+mostly)/total, float(full+mostly+partial)/total, float(full+mostly+partial+near)/total, float(full+mostly+partial+near+miss)/total]
                    report.write('Recall:\t\t'+ViolaDetector.get_formatted_floats(float_list))
                    evaluation[roof_type][coverage_type]['recall'] = float_list

                    #Precision: get the Number of detections 
#                    report.write('Precision: \t\t{0:.{s}f}\t\t{1:.{s}f}\t\t{2:.{s}f}\t\t{3:.{s}f}\t\t{4:.{s}f}\n'.format(float(full)/detection_num, float(full+mostly)/detection_num, 
#                        float(full+mostly+partial)/detection_num, float(full+partial+near)/detection_num,float(full+partial+near+miss)/detection_num), s=4)
                    float_list = [float(full)/detection_num,float(full+mostly)/detection_num, float(full+mostly+partial)/detection_num, 
                                    float(full+mostly+partial+near)/detection_num, float(full+mostly+partial+near+miss)/detection_num]
                    report.write('Precis:\t\t'+ViolaDetector.get_formatted_floats(float_list))
                    evaluation[roof_type][coverage_type]['precision'] = float_list

                    #False alarm rate = Detections-roofs_detected/detections
                    float_list = [float(detection_num-full)/detection_num,float(detection_num-(full+mostly))/detection_num, float(detection_num-(full+mostly+partial))/detection_num, 
                                    float(detection_num-(full+mostly+partial+near))/detection_num, float(detection_num-(full+mostly+partial+near+miss))/detection_num]
                    report.write('F.Alarm:\t'+ViolaDetector.get_formatted_floats(float_list)) 
                    evaluation[roof_type][coverage_type]['false_alarm'] = float_list

                report.write('\n\n\n')

        #pickle the evaluation
        with open(self.output_folder+'evaluation.pickle', 'wb') as f:
            pickle.dump(evaluation, f)


    @staticmethod
    def get_formatted_floats(float_list):
        formatted_list = list()
        for f in float_list:
            formatted_list.append("{:5.4f}".format(f))
        return '\t\t'.join(formatted_list)+'\n'


    def mark_detections_on_img(self, img, img_name):
        '''Return an image with the detections and the ground truth roofs marked with rectangles
        '''
        for i, roof_type in enumerate(self.viola_detections.roof_types):
            for (x,y,w,h) in (self.viola_detections.get_img_detections_specific_type(roof_type, img_name)):
                if i%1 == 0:
                    color=(0,0,255)
                else:
                    color=(255,0,0)
                cv2.rectangle(img,(x,y),(x+w,y+h), color, 2)
        return img


    def mark_false_true_detections_and_roofs(self, out_path, img_path, img_name, roof_list, false_pos, true_pos):
        #get the img
        try:
            img = cv2.imread(img_path+img_name, flags=cv2.IMREAD_COLOR)
        except:
            print 'Cannot open {0}'.format(img_path+img_name)

        img = self.mark_roofs_on_img(img, img_name, roof_list, (255,0,0))
        img = self.mark_roofs_on_img(img, img_name, false_pos, (0,255,0))
        img = self.mark_roofs_on_img(img, img_name, true_pos, (255,255,255))

        #save image 
        cv2.imwrite(out_path+img_name+'_FP_TP.jpg', img) 

    def mark_roofs_on_img(self, img, img_name, roof_list, color=(0,255,0)):
        for roof in roof_list:
            if type(roof) is Roof:
                x, y, w, h = roof.xmin, roof.ymin, roof.width, roof.height
            else:
                x, y, w, h = roof
            cv2.rectangle(img,(x,y),(x+w, y+h),color,2)
        return img


    def save_detection_img(self, img_path, img_name, roof_list):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.mark_detections_on_img(img, img_name)
        img = self.mark_roofs_on_img(img, img_name, roof_list)
        self.save_image(img, img_name)


    def save_image(self, img, img_name, output_folder=None):
        self.output_folder = output_folder if output_folder is not None else self.output_folder  
        cv2.imwrite(self.output_folder+'_'+img_name, img)


def pickle_neural_true_false_positives():
    #Getting patches for neural network
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-c':
            combo_f = arg
    assert combo_f is not None
    detectors = get_detectors(combo_f)
    viola = ViolaDetector(load_pickle_stats = False, detector_names=detectors, save_imgs=False, old_detector = False, neural=True)
    if len(detectors['metal']) > 0:
        roof_type = 'metal'
    elif len(detectors['thatch']) > 0:
        roof_type = 'thatch'
    else:
        raise ValueError('No detector found in combo{0}'.format(combo_f))
    viola.get_neural_training_data(save_false_pos = True, roof_type=roof_type, detector_name='combo'+combo_f) 


def get_detectors(combo_f):
    detectors = dict()
    detector_file = '../viola_jones/detector_combos/combo'+str(combo_f)+'.csv'
    detectors = defaultdict(list)
    with open(detector_file, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for line in r:
            if len(line) < 2:
                continue
            if line[0] == 'metal':
                detectors['metal'].append(line[1].strip())
            elif line[0] == 'thatch':
                detectors['thatch'].append(line[1].strip())
            else:
                raise ValueError("Unknown detector type {0}".format(line[0]))
    return detectors


def get_all_combos():
    path = settings.COMBO_PATH
    detector_list = list()
    combo_f_names = list()

    for f_name in os.listdir(path):
        if os.path.isfile(path+f_name) and f_name.startswith('combo') and ('equalized') in f_name:
            combo_f = f_name[5:7]
            if combo_f.endswith('.'):
                combo_f = combo_f[:1]
            
            detector_list.append(get_detectors(combo_f))
            combo_f_names.append(combo_f)
    return detector_list, combo_f_names 

def testing_detectors(original_dataset=False, all=False, validation=False, small_test=False):
    '''Test either a single detector or all detectors in the combo files
    '''
    #TESTING ONLY ONE DETECTOR that must be passed in as an argument  
    if all == False:
        #combo_f = raw_input('Enter detector combo file number: ')
        combo_f = None
        try:
            opts, args = getopt.getopt(sys.argv[1:], "c:")
        except getopt.GetoptError:
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-c':
                combo_f = arg
        assert combo_f is not None
        detectors = get_detectors(combo_f)
        detector_list = [detectors]
        combo_f_names = [combo_f]
    else:#get all detector combos
        detector_list, combo_f_names =  get_all_combos()

    #need to detect on validation set. To make it easier to digest, initially we also looked at training set
    for detector, combo_f_name in zip(detector_list, combo_f_names):
        if original_dataset == False:
            if validation and small_test==False:
                out_path = settings.VIOLA_OUT+'with_validation_set/'
                in_path = settings.VALIDATION_PATH 
            elif validation == False and small_test ==False:
                out_path = settings.VIOLA_OUT+'with_training_set/'
                in_path = settings.TRAINING_PATH
            else:
                out_path = settings.VIOLA_OUT+'small_test/'
                in_path = '../data/small_test/'
        else:
            out_path = settings.ORIGINAL_VIOLA_OUTPUT
            in_path = settings.ORIGINAL_VALIDATION_PATH
        folder_name = 'combo'+combo_f_name+'/'
        viola = ViolaDetector(load_pickle_stats = False, folder_name=folder_name, out_path=out_path, detector_names=detector, save_imgs=False, old_detector = False)
        pdb.set_trace()
        viola.detect_evaluate_roofs_folder(path=in_path, save_detections=True, reject_levels=0.5, level_weights=2, scale=1.05)



def check_cascade_status():
    '''Check the status of all of the cascades
    '''
    casc_path = '../viola_jones/'
    for f_name in os.listdir(casc_path):
        if os.path.isdir(casc_path+f_name) and f_name.startswith('cascade_'):
            if os.path.isfile(casc_path+f_name+'/cascade.xml'):
                print '{0}\t\t\t done'.format(f_name)
            else:
                print '{0}\t\t\t MISSING'.format(f_name)


def set_up_basic_combos():
    '''Set up combo with a single detector per combo
    '''
    casc_path = '../viola_jones/'
    out_path = '../viola_jones/detector_combos/combo'
    combo_num = 0
    f_names = dict() #roof_type, equalized, augmented
    f_names['metal'] = dict()
    f_names['thatch'] = dict()
    f_names['metal']['equalized']=defaultdict(set)
    f_names['metal']['not_equalized'] =defaultdict(set)
    f_names['thatch']['equalized']=defaultdict(set)
    f_names['thatch']['not_equalized']=defaultdict(set)

    for f_name in os.listdir(casc_path):
        if os.path.isdir(casc_path+f_name) and f_name.startswith('cascade_') and ('equalized' in f_name):
            if os.path.isfile(casc_path+f_name+'/cascade.xml'):
                try:
                    roof_type = 'metal' if 'metal' in f_name else 'thatch'
                    equalized = 'not_equalized' if 'not_equalized' not in f_name else 'equalized'
                    augmented = 'augm1' if 'augm1' in f_name else 'augm0'
                    f_names[roof_type][equalized][augmented].add(f_name)
                except KeyError:
                    pdb.set_trace()
                detector_name = f_name[8:]
                with open('{0}{1}.csv'.format(out_path, combo_num), 'w') as f:
                    if detector_name.startswith('metal'):
                        d_type = 'metal'
                    elif detector_name.startswith('thatch'):
                        d_type = 'thatch'
                    else:
                        raise ValueError('Unknown roof type for cascade')

                    f.write('{0}, {1}'.format(d_type, detector_name))
                combo_num += 1
            else:
                print 'Could not process incomplete: {0}'.format(f_name)
    #set up the rectangular and square detectors together 
    for roof_type in ['metal', 'thatch']:
        for equalized in ['equalized', 'not_equalized']:
            for augm in ['augm1', 'augm0']:
                detectors = f_names[roof_type][equalized][augm]
                if len(detectors) > 1:
                    #write a new combo file
                    with open('{0}{1}.csv'.format(out_path, combo_num), 'w') as f:
                        log_to_file = ''
                        for d in detectors:
                            detector_name = d[8:]
                            if detector_name.startswith('metal'):
                                d_type = 'metal'
                            elif detector_name.startswith('thatch'):
                                d_type = 'thatch'
                            else:
                                raise ValueError('Unknown roof type for cascade')
                            log_to_file += '{0}, {1}\n'.format(d_type, detector_name)
                        f.write(log_to_file)
                        combo_num += 1
                else:
                    print 'Only one detector found: {0}'.format(detectors)


def get_img_size():
    for i, f_name in enumerate(os.listdir(settings.TRAINING_PATH)):
        img = cv2.imread(settings.TRAINING_PATH+f_name)
        print img.shape[0], img.shape[1]
        
        if i%20==0:
            pdb.set_trace()




if __name__ == '__main__':
#    check_cascade_status()
#    set_up_basic_combos()
#    get_img_size()
    testing_detectors(all=False, original_dataset=True, validation=True)
    
#    pickle_neural_true_false_positives()
