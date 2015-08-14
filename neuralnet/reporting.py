from collections import defaultdict
import numpy as np
from os import listdir
import sys
import pdb
import cv2
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt

from get_data import DataLoader #for get_roofs
import utils


class Detections(object):
    def __init__(self, roof_types=None, mergeFalsePos=False):
        self.total_detection_num = defaultdict(int)
        self.false_positive_num =  defaultdict(int)
        self.true_positive_num =  defaultdict(int)
        self.good_detection_num =  defaultdict(int)

        self.easy_false_neg_num = defaultdict(int)
        self.easy_false_pos_num = defaultdict(int)
        self.easy_true_pos_num = defaultdict(int) 

        if mergeFalsePos:
            self.bad_detection_num = 0 #we count the metal and thatch false positives together 
        else:
            self.bad_detection_num = defaultdict(int)
        self.roof_num =  defaultdict(int)

        self.detections = defaultdict(list)
        self.true_positives = dict()
        self.false_positives = dict()
        self.good_detections =dict()#all the detections above the VOC threshold

        #this metrics do not have double detections count negatively, and
        #they consider a roof to have been detected if a detection has 'mostly' roof on it
        self.easy_false_pos = dict()
        self.easy_false_neg = dict()
        self.easy_true_pos = dict()

        #index with the roof polygon coordinates
        #stores the best score and detection for each grount true roof
        self.roof_detections_voc = dict() 
        self.best_score_per_detection = dict() #keep track of each detection's best voc score with some roof, for each image

        if mergeFalsePos:
            self.bad_detections = defaultdict(list) #there's only one set of bad detections
        else:
            self.bad_detections = dict()
        for roof_type in utils.ROOF_TYPES: # we need this to separate them per image
            self.roof_detections_voc[roof_type] = defaultdict(list)
            self.detections[roof_type] = dict()
            self.true_positives[roof_type] = defaultdict(list)
            self.false_positives[roof_type] = defaultdict(list)
            self.good_detections[roof_type] = defaultdict(list)

            if mergeFalsePos == False:
                self.bad_detections[roof_type] = defaultdict(list)

            #more lenient metrics
            self.easy_false_pos[roof_type] = defaultdict(list)
            self.easy_false_neg[roof_type] = defaultdict(list)
            self.easy_true_pos[roof_type] = defaultdict(list)
        self.total_time = 0
        self.imgs = set()

    def set_best_voc(self,img_name=None, roof_type=None, roof_polygon=None, best_detection=None, score=None):
        self.roof_detections_voc[roof_type][img_name].append((roof_polygon, best_detection, score))  

    def update_true_pos(self, true_pos=None, img_name=None, roof_type=None):
        self.true_positives[roof_type][img_name].extend(true_pos)
        self.true_positive_num[roof_type] += len(true_pos)

    def update_false_pos(self, false_pos=None, img_name=None, roof_type=None):
        self.false_positives[roof_type][img_name].extend(false_pos)
        self.false_positive_num[roof_type] += len(false_pos)

    def update_good_detections(self, good_detections=None, img_name=None, roof_type=None):
        self.good_detections[roof_type][img_name].extend(good_detections)
        self.good_detection_num[roof_type] += len(good_detections)

    def update_bad_detections(self, roof_type=None, bad_detections=None, img_name=None):
        if roof_type is not None:
            self.bad_detections[roof_type][img_name].extend(bad_detections)
            self.bad_detection_num[roof_type] += len(bad_detections)
        else:
            self.bad_detections[img_name].extend(bad_detections)
            self.bad_detection_num += len(bad_detections)

    def update_roof_num(self, roofs, roof_type):
        self.roof_num[roof_type] += len(roofs)

    def set_detections(self, roof_type=None, img_name=None, angle=None, detection_list=None, img=None):
        '''@param img is the rotated image
        '''
        #if we don't pass in an angle, we simply set everything to be from angle zero
        angle = angle if angle is not None else 0

        #ensure we have a place to store detections for all angles for the current image/rooftype
        if img_name not in self.detections[roof_type]:
            self.detections[roof_type][img_name] = defaultdict(list)

        #update the detection count for this roof type
        self.total_detection_num[roof_type] += len(detection_list)
        self.detections[roof_type][img_name][angle].extend(detection_list)


    def get_detections(self, img_name=None, angle=None, roof_type=None):
        assert img_name is not None
        detections = list()

        if roof_type is None:   #any roof_type
            if angle is None:
                #return all detections regardless of angle or rooftype
                for roof_type in self.detections.keys():
                    if img_name in self.detections[roof_type]:
                        for angle in self.detections[roof_type][img_name].keys():
                            detections.extend(self.detections[roof_type][img_name][angle])

            else:
                #return detections for some angle
                for roof_type in utils.ROOF_TYPES:
                    detections.extend(self.detections[roof_type][img_name][angle])

        else:   #specific roof_type
            if angle is None:
                #return detections for this roof type
                if img_name in self.detections[roof_type]:
                    for angle in self.detections[roof_type][img_name].keys():
                        detections.extend(self.detections[roof_type][img_name][angle])
            else:
                #turn detections for this roof type AND angle
                detections = self.detections[roof_type][img_name][angle]
        return detections


    def update_easy_metrics(self, img_name, roof_type, easy_false_neg, easy_true_pos, easy_false_pos):
        self.easy_false_neg[roof_type][img_name].extend(easy_false_neg)
        self.easy_true_pos[roof_type][img_name].extend(easy_true_pos)
        self.easy_false_pos[roof_type][img_name].extend(easy_false_pos) 

        self.easy_false_neg_num[roof_type] += len(easy_false_neg)
        self.easy_false_pos_num[roof_type] += len(easy_false_pos)
        self.easy_true_pos_num[roof_type] += len(easy_true_pos)
        



class Evaluation(object):
    def __init__(self, report_name=None, method=None, 
                        full_dataset=True, 
                        folder_name=None, save_imgs=True, out_path=None, 
                        detections=None, in_path=None, 
                        detector_names=None, 
                        mergeFalsePos=False,
                        separateDetections=True,
                        vocGood=0.1, negThres = 0.3):
        '''
        Will score the detections class it contains.

        Parameters:
        --------------------
        separateDetections: bool
            Whether the metal detections can count as true positives for the thatch 
            detections and viceversa
        mergeFalsePos: bool
            Whether we need to keep track of the good and bad detections of metal
            and thatch separately or not. We cannot separate them if we want to 
            train a single neural network to distinguish between both types of roofs 
            since the bad detections of either roof must not contain a positive detection
            of the other type of roof (it should be background)
        '''
        self.TOTAL = 0
        self.save_imgs = save_imgs
        #these two are related to saving the FP and TP for neural training
        self.mergeFalsePos=mergeFalsePos

        self.keep_detections_separate = separateDetections  

        #threholds to classify detections as False/True positives and Good/Bad detections(these are to train the neural network on it)
        self.VOC_threshold = utils.VOC_threshold #threshold to assign a detection as a true positive
        self.VOC_good_detection_threshold = dict()
        self.VOC_good_detection_threshold['metal'] = utils.VOC_threshold 
        self.VOC_good_detection_threshold['thatch'] = utils.VOC_threshold
        self.detection_portion_threshold = 0.50
        self.negThres = negThres

        self.detections = detections    #detection class

        self.in_path = in_path          #the path from which the images are taken
        self.img_names = [f for f in listdir(self.in_path) if f.endswith('.jpg')]

        #the ground truth roofs for every image and roof type
        self.correct_roofs = dict()
        for roof_type in utils.ROOF_TYPES:
            self.correct_roofs[roof_type] = dict()
        self.full_dataset = full_dataset
        if full_dataset == False:
            for img_name in self.img_names:
                for roof_type in utils.ROOF_TYPES:
                    self.correct_roofs[roof_type][img_name] = DataLoader.get_polygons(roof_type=roof_type, 
                                                                xml_name=img_name[:-3]+'xml' , xml_path=self.in_path)
                    self.detections.update_roof_num(self.correct_roofs[roof_type][img_name], roof_type)
        else:
            for img_name in self.img_names:
                current_roofs = DataLoader.get_all_roofs_full_dataset(xml_name=img_name[:-3]+'xml' , xml_path=self.in_path)
                for roof_type in utils.ROOF_TYPES:
                    if roof_type not in current_roofs:
                        self.correct_roofs[roof_type][img_name] = []
                    else:
                        self.correct_roofs[roof_type][img_name] = current_roofs[roof_type] 

                    self.detections.update_roof_num(self.correct_roofs[roof_type][img_name], roof_type)
 

        #init the report file
        self.out_path = out_path
        if detector_names is not None:
            self.init_report(detector_names, report_name=report_name)

        #variables needed to pickle the FP and TP to a file that makes sense
        assert method is not None
        self.method = method
        self.folder_name = folder_name
        #self.datasource = self.set_datasource_name(in_path)


    def init_report(self, detector_names, report_name=None):
        report_name = report_name if report_name is not None else 'report.txt'
        self.report_file = self.out_path+report_name
        try:
            report = open(self.report_file, 'w')
            report.write('metal detectors: \t')
            report.write('\t'.join(detector_names['metal']))
            report.write('\n')
            report.write('thatch_detector: \t')
            report.write('\t'.join(detector_names['thatch']))           
            report.write('\n')
        except IOError as e:
            print e
        else:
            report.close()

    
    def score_img(self, img_name, img_shape, contours=False, fast_scoring=False):
        '''Find best overlap between each roof in an img and the detections,
        according the VOC score
        '''
        print 'Scoring {}.....'.format(img_name)
        #start with all detections as false pos; as we find matches, we increase the true_pos, decrese the false_pos numbers
        false_pos_logical = dict()
        bad_detection_logical = dict()
        best_score_per_detection = defaultdict(list) #for the current image
        detections = dict()

        easy_false_pos_logical = dict() #keeps track of the detections that are VOC>0.5 or cover mostly a roof
        easy_false_negative_logical=dict()

        for roof_type in utils.ROOF_TYPES:
            #this will only be necessary if we work with the neural network
            #and we want to score roofs by merging nearby detections
            work_with_contours = False
            if roof_type=='metal' and contours:
                work_with_contours = True

            #we may want to keep all detections together, but this lowers precision a lot
            #by default, metal and thatch detections are evaluated separately
            if self.keep_detections_separate:
                detections[roof_type] = self.detections.get_detections(img_name=img_name, roof_type=roof_type)
            else:
                detections[roof_type] = self.detections.get_detections(img_name=img_name)

            print 'Scoring {0}'.format(roof_type)
            #detections that are wrong accoring to VOC metric
            false_pos_logical[roof_type] = np.ones(len(detections[roof_type]), dtype=bool) #[roof_type]
            #this is used to get the training data for TP and FP
            bad_detection_logical[roof_type] = np.ones(len(detections[roof_type]), dtype=bool) #[roof_type]

            #how many detections are wrong
            easy_false_pos_logical[roof_type] = np.ones(len(detections[roof_type]), dtype=bool)
            #how many roofs we have missed
            easy_false_negative_logical[roof_type] = np.ones(len(self.correct_roofs[roof_type][img_name]), 
                                                            dtype=bool)

            for r, roof in enumerate(self.correct_roofs[roof_type][img_name]):

                best_voc_score = -1 
                best_detection = -1 

                for d, detection in enumerate(detections[roof_type]):                             #for each patch found
                    if d%1000 ==0:
                        #print 'Roof {}/{} Detection {}/{}'.format(r, len(self.correct_roofs[roof_type][img_name]), d, len(detections[roof_type]))
                        pass
                    if r == 0:#first roof, so insert the current detection and a negative score
                        best_score_per_detection[roof_type].append([detection, -1])

                    #detection_roof_portion is how much of the detection is covered by roof
                    if fast_scoring:
                         voc_score, detection_roof_portion = self.get_score_fast(roof=roof, detection=detection)
                    else:
                        voc_score, detection_roof_portion = self.get_score(contours=work_with_contours, 
                                                        rows=img_shape[0], cols=img_shape[1], roof=roof, detection=detection)
                    if (voc_score > self.VOC_threshold) and (voc_score > best_voc_score):#this may be a true pos
                        best_voc_score = voc_score
                        best_detection = d 
                    
                    if (voc_score > self.VOC_threshold) or (detection_roof_portion > 0.5):
                        easy_false_pos_logical[roof_type][d] = 0
                        easy_false_negative_logical[roof_type][r] = 0 

                    if (voc_score > self.VOC_good_detection_threshold[roof_type]): # or detection_roof_portion>self.detection_portion_threshold):
                        bad_detection_logical[roof_type][d] = 0 #we already know that this wasn't a bad detection

                    if (voc_score > best_score_per_detection[roof_type][d][1]): #keep track of best match with some roof for each detection
                        best_score_per_detection[roof_type][d][1] = voc_score
                        
                    #store the best detection for this roof regardless of whether it is over 0.5
                    if voc_score > best_voc_score:
                        self.detections.set_best_voc(img_name=img_name, roof_type=roof_type, 
                                            roof_polygon=roof, best_detection=detections[roof_type][d], score=best_voc_score)  

                if best_detection != -1:
                    false_pos_logical[roof_type][best_detection] = 0 #this detection is not a false positive

        self.update_scores(img_name, detections, false_pos_logical, bad_detection_logical, 
                                    best_score_per_detection, easy_false_pos_logical, easy_false_negative_logical)
        self.save_images(img_name)

       

    def update_scores(self, img_name, detections, false_pos_logical, bad_detection_logical, 
                                                best_score_per_detection, easy_false_pos_logical, easy_false_negative_logical):
        self.detections.best_score_per_detection[img_name] = best_score_per_detection

        if self.mergeFalsePos: #self.output_patches 
            detects = np.array(detections['metal'].extend(detections['thatch']))
            #a detection is only considered bad if it doesn't match either a metal or a thatch roof
            truly_bad_detections = np.logical_and(bad_detection_logical['metal'], bad_detection_logical['thatch']) 
            bad_d = detects[truly_bad_detections]
            self.detections.update_bad_detections(bad_detections=bad_d, img_name=img_name) 

        print '----- METHOD: {}'.format(self.method)
        for roof_type in utils.ROOF_TYPES: 
            detects = np.array(detections[roof_type])
            false_d = detects[false_pos_logical[roof_type]]
            self.detections.update_false_pos(false_pos=false_d, roof_type=roof_type, img_name=img_name) 
            pos_d = detects[np.invert(false_pos_logical[roof_type])]
            self.detections.update_true_pos(true_pos=pos_d, roof_type=roof_type, img_name=img_name) 
            
            #if self.output_patches:
            #get good are good for each specific roof type separately, unlike the truly bad detections above 
            good_d = detects[np.invert(bad_detection_logical[roof_type])]
            self.detections.update_good_detections(good_detections=good_d, roof_type=roof_type, img_name=img_name) 

            #NEW METRICS ADDED THAT ARE MORE LENIENT
            roofs = np.array(self.correct_roofs[roof_type][img_name])
            #how many roofs we have missed
            easy_false_neg = roofs[easy_false_negative_logical[roof_type]]
            #how many roofs we found
            easy_true_pos = roofs[np.invert(easy_false_negative_logical[roof_type])]
            #how many bad detections
            easy_false_pos = detects[easy_false_pos_logical[roof_type]] 
            #the true neg are everything that was detected as background... we don't care much about this?
            self.detections.update_easy_metrics(img_name, roof_type, easy_false_neg, easy_true_pos, easy_false_pos)


            if self.mergeFalsePos == False:
                bad_d = detects[bad_detection_logical[roof_type]]
                self.detections.update_bad_detections(roof_type=roof_type,bad_detections=bad_d, img_name=img_name) 
            
            print '-------- Roof Type: {0} --------'.format(roof_type)
            print 'Roofs: {0}'.format(len(self.correct_roofs[roof_type][img_name]))
            print 'False pos: {0}'.format(len(false_d))
            print 'True pos: {0}'.format(len(pos_d))
            #if self.output_patches:
            print 'Good det: {0}'.format(len(good_d))
            if self.mergeFalsePos == False:
                print 'Bad detections: {0}'.format(len(bad_d))
            print 'EASIER METRICS'
            print 'True pos: {}'.format(len(easy_true_pos))
            print 'False neg: {}'.format(len(easy_false_neg))
            print 'False pos: {}'.format(len(easy_false_pos))

        print '---'
        print 'All Detections: {0}'.format(len(detections['metal']+detections['thatch']))
        if self.mergeFalsePos: #self.output_patches 
            print 'All Bad det: {0}'.format(len(bad_d))


    def get_score(self, contours=False, rows=1200, cols=2000, roof=None, detection=None):
        #assert len(roof) == 4 and len(detection) == 4

        #First, count the area that was detected
        count_detection_area_mask = np.zeros((rows, cols), dtype='uint8')
        if contours == False:
            utils.draw_detections(np.array([detection]), count_detection_area_mask, fill=True, color=1)
        else:
            cv2.drawContours(count_detection_area_mask, np.array([detection]), 0, 1, -1)
        detection_area = utils.sum_mask(count_detection_area_mask)

        #binary mask of ground truth roof on the original image, non-rotated
        
        matching_mask = np.zeros((rows, cols), dtype='uint8')
        utils.draw_detections(np.array([roof]), matching_mask, fill=True, color=1)
        roof_area = utils.sum_mask(matching_mask)

        #subtract the detection
        if contours == False:
            utils.draw_detections(np.array([detection]), matching_mask, fill=True, color=0)
        else:
            cv2.drawContours(matching_mask, [detection], 0, 0, -1)

        #calculate the intersection 
        roof_missed = utils.sum_mask(matching_mask)
        intersection_area = roof_area-roof_missed

        #VOC measure
        union_area = (roof_area + (detection_area)) - intersection_area
        voc_score = float(intersection_area)/union_area

        #How much of the detection is roof? If it's high, they this detection is mostly covering a roof
        detection_roof_portion = float(intersection_area)/detection_area 
        return voc_score, detection_roof_portion


    def get_score_fast(self, roof, detection):
        '''Can use this method if we don't have rotated rectangles
        '''
        intersection_area = 0
        roof_xmin, roof_ymin, roof_xmax, roof_ymax = roof
        detection_xmin, detection_ymin, detection_xmax, detection_ymax = detection

        dx = min(roof_xmax, detection_xmax) - max(roof_xmin, detection_xmin)
        dy = min(roof_ymax, detection_ymax) - max(roof_ymin, detection_ymin)
        if (dx>=0) and (dy>=0):
            intersection_area = dx*dy

        #VOC measure
        roof_area = (roof_xmax - roof_xmin) * (roof_ymax - roof_ymin)
        detection_area = (detection_xmax - detection_xmin) * (detection_ymax - detection_ymin)
        union_area = (roof_area + detection_area) - intersection_area
        voc_score = float(intersection_area)/union_area

        #How much of the detection is roof? If it's high, they this detection is mostly covering a roof
        detection_roof_portion = float(intersection_area)/detection_area 
        return voc_score, detection_roof_portion

       

    def print_report(self, print_header=True, stage=None, report_name='report.txt', write_to_file=True):
        '''
        Parameters:
        ---------------
        stage: string
            can add an additional column to the report for instance, if theres' multiple stages that we want to report in a single file,
            as is the case of the pipeline detection
        '''
        log_to_file = list() 
        if print_header: 
            open(self.out_path+report_name, 'w').close() 
            if stage is None:
                log_to_file.append('roof_type\ttotal_roofs\ttotal_time\tdetections\trecall\tprecision\tf1')
            else:
                log_to_file.append('stage\troof_type\ttotal_roofs\ttotal_time\tdetections\trecall\tprecision\tf1')
            easy_log.append('roof_type\ttotal_roofs\ttotal_time\tdetections\trecall\tprecision\tf1')


        for roof_type in utils.ROOF_TYPES:
            detection_num = self.detections.total_detection_num[roof_type]
            true_pos = self.detections.true_positive_num[roof_type]
            false_pos = self.detections.false_positive_num[roof_type]
            cur_type_roofs = self.detections.roof_num[roof_type] 

            if detection_num > 0 and cur_type_roofs > 0:
                recall = float(true_pos) / cur_type_roofs 
                precision = float(true_pos) / detection_num
                if precision+recall > 0:
                    F1 = (2.*precision*recall)/(precision+recall)
                else:
                    F1 = 0
            else:
                recall = precision = F1 = 0
            if stage is None:
                log_to_file.append('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(roof_type, 
                                cur_type_roofs, self.detections.total_time, detection_num, recall, precision,F1)) 
            else:
                log_to_file.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(stage, roof_type, 
                                cur_type_roofs, self.detections.total_time, detection_num, recall, precision,F1)) 

            #EASIER METRICS
            easy_true_pos = self.detections.easy_true_pos_num[roof_type]
            easy_false_pos = self.detections.easy_false_pos_num[roof_type]
            easy_detection_num = easy_true_pos+easy_false_pos
            if easy_detection_num > 0 and cur_type_roofs > 0:
                easy_recall = float(easy_true_pos) / cur_type_roofs 
                easy_precision = float(easy_true_pos) / (easy_detection_num)
                if precision+recall > 0:
                    F1 = (2.*easy_precision*easy_recall)/(easy_precision+easy_recall)
                else:
                    F1 = 0
            else:
                recall = precision = F1 = 0
            easy_log.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(stage, roof_type, 
                        cur_type_roofs, self.detections.total_time, detection_num, easy_recall, easy_precision,easy_F1))

        log = '\n'.join(log_to_file)
        easy_log = '\n'.join(easy_log)
        with open(self.out_path+report_name, 'a') as report:
            report.write(log)
        print 'FINAL REPORT'
        print log
        print easy_log
 



    def save_images(self, img_name, fname=''):
        '''Displays the ground truth, along with the true and false positives for a given image
        '''
        try:
            img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
        except IOError:
            print 'Cannot open {0}'.format(self.in_path+img_name)
            sys.exit(-1)

        rects = True if self.full_dataset else False
        #for roof_type in utils.ROOF_TYPES:
        #    utils.draw_detections(self.detections.false_positives[roof_type][img_name], img, color=(0, 0, 255), rects=rects)
        for roof_type in utils.ROOF_TYPES:
            utils.draw_detections(self.correct_roofs[roof_type][img_name], img, color=(255, 0, 0), rects=rects)
        for roof_type in utils.ROOF_TYPES:
            utils.draw_detections(self.detections.true_positives[roof_type][img_name], img, color=(0, 255, 0), rects=rects)


        cv2.imwrite('{0}{1}{2}.jpg'.format(self.out_path, img_name[:-4], fname), img)



    def get_patch_mask(self, rows=None, cols=None, detections=None):
        '''Mark the area of a detection as True, everything else as False
        '''
        assert rows is not None
        assert cols is not None

        patch_mask = np.zeros((rows, cols), dtype=bool)
        patch_area = 0
        for (x,y,w,h) in detections:
            patch_mask[y:y+h, x:x+w] = True
            patch_area += w*h

        area_covered = np.sum(patch_area)
        return patch_mask, float(area_covered)/(rows*cols) 


    def get_bounding_rects(self, img_name=None, rows=None, cols=None, detections=None):
        mask, _ = self.get_patch_mask(rows, cols, detections)
        mask_img = np.array(mask, dtype=int)
        mask_path = 'mask.jpg'
        cv2.imwrite(mask_path, mask_img) 
        im_gray = cv2.imread(mask_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #find and save bouding rectangles of contours
        output_bounds = np.zeros((im_bw.shape[0],im_bw.shape[1]))
        bounding_rects = list()
        for cont in contours:
            bounding_rect = cv2.boundingRect(cont)
            bounding_rects.append(bounding_rect)
            cv2.rectangle(output_bounds, (bounding_rect[0],bounding_rect[1]),(bounding_rect[0]+bounding_rect[2],
                                bounding_rect[1]+bounding_rect[3]), (255,255,255),5)
        #write to file
        if self.save_imgs:
            bounding_path = self.out_path+img_name+'_bound.jpg'
            cv2.imwrite(bounding_path, output_bounds)
        return bounding_rects


    def save_training_TP_FP_using_voc(self, rects=False, neural=True, viola=False, img_names=None):
        '''use the voc scores to decide if a patch should be saved as a TP or FP or not
        '''
        general_path = utils.get_path(neural=neural, viola=viola, data_fold=utils.TRAINING, in_or_out=utils.IN, out_folder_name=self.folder_name)
        general_path = '../training_data_full_dataset_neural/{}'.format(self.folder_name)
        utils.mkdir(out_folder_path=general_path)
        
        path_true = general_path+'truepos/'
        utils.mkdir(path_true)

        path_false = general_path+'falsepos/'
        utils.mkdir(path_false)
        img_names = img_names if img_names is not None else self.img_names

        num_patches = 0 #we can only save around 30000 images per folder!!!
        for i, img_name in enumerate(img_names):
            print 'Saving patches for {} {}/{}'.format(img_name, i+1, len(img_names))

            good_detections = defaultdict(list)
            bad_detections = defaultdict(list)
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

            for roof_type in utils.ROOF_TYPES:
                detection_scores = self.detections.best_score_per_detection[img_name][roof_type]
                for detection, score in detection_scores:
                    if score > 0.5:
                        #true positive
                        good_detections[roof_type].append(detection)
                    if score < self.negThres:
                        #false positive
                        bad_detections[roof_type].append(detection)
                    
            for roof_type in utils.ROOF_TYPES:
                extraction_type = 'good'
                num_patches = self.save_training_FP_and_TP_helper(num_patches, img_name, good_detections[roof_type], path_true, 
                                                    general_path, img, roof_type, extraction_type, (0,255,0), rects=rects)               
                extraction_type = 'background'
                num_patches = self.save_training_FP_and_TP_helper(num_patches, img_name, bad_detections[roof_type], path_false, 
                                                    general_path, img, roof_type, extraction_type, (0,0,255), rects=rects)               


    def save_training_FP_and_TP_helper(self, num_patches, img_name, detections, patches_path, 
                                        general_path, img, roof_type, extraction_type, color, rects=False):
        #this is where we write the detections we're extraction. One image per roof type
        #we save: 1. the patches and 2. the image with marks of what the detections are, along with the true roofs (for debugging)
        img_debug = np.copy(img) 
        if roof_type == 'background':
            utils.draw_detections(self.correct_roofs['metal'][img_name], img_debug, color=(0, 0, 0), thickness=2, rects=rects)
            utils.draw_detections(self.correct_roofs['thatch'][img_name], img_debug, color=(0, 0, 0), thickness=2, rects=rects)
        else:
            utils.draw_detections(self.correct_roofs[roof_type][img_name], img_debug, color=(0, 0, 0), thickness=2, rects=rects)

        for i, detection in enumerate(detections):
            batch_path =  'batch{}/'.format(int(num_patches/20000))  
            if num_patches % 20000 == 0:
                utils.mkdir('{}falsepos/batch{}/'.format(general_path, num_patches/20000))
                utils.mkdir('{}truepos/batch{}/'.format(general_path, num_patches/20000))
            num_patches += 1

            current_patch_path = patches_path+batch_path

            #extract the patch, rotate it to a horizontal orientation, save it
            if rects == False:
                bitmap = np.zeros((img.shape[:2]), dtype=np.uint8)
                padded_detection = utils.add_padding_polygon(detection, bitmap)
                warped_patch = utils.four_point_transform(img, padded_detection)
                cv2.imwrite('{0}{1}_{2}_roof{3}.jpg'.format(current_patch_path, roof_type, img_name[:-4], i), warped_patch)
                
                #mark where roofs where taken out from for debugging
                utils.draw_polygon(padded_detection, img_debug, fill=False, color=color, thickness=2, number=i)
            else:
                pad = 10
                xmin = (detection.xmin-pad) if (detection.xmin-pad)>0 else detection.xmin
                ymin = (detection.ymin-pad) if (detection.ymin-pad)>0 else detection.ymin
                xmax = (detection.xmax+pad) if (detection.xmax+pad)<img.shape[1] else detection.xmax
                ymax = (detection.ymax+pad) if (detection.ymax+pad)<img.shape[0] else detection.ymax
                patch = img[ymin:ymax, xmin:xmax, :]
                #print 'saving {0}{1}_{2}_roof{3}.jpg'.format(current_patch_path, roof_type, img_name[:-4], i) 
                cv2.imwrite('{0}{1}_{2}_roof{3}.jpg'.format(current_patch_path, roof_type, img_name[:-4], i), patch)
                self.TOTAL += 1
                if self.TOTAL % 1000 == 0:
                    print 'Saved {} patches'.format(self.TOTAL)
        return num_patches
        #write this type of extraction and the roofs to an image
        #cv2.imwrite('{0}{1}_{2}_extract_{3}.jpg'.format(general_path, img_name[:-4], roof_type, extraction_type), img_debug)


    def auc_plot(self,correct_classes, class_probs):
        prec = dict()
        recall = dict()
        average_precision = dict()
        for roof_type in utils.ROOF_TYPES:
            prec[roof_type], recall[roof_type], _ = precision_recall_curve(np.array(correct_classes[roof_type]),
                                                                        np.array(class_probs[roof_type]))
            average_precision[roof_type] = average_precision_score(np.array(correct_classes[roof_type]),
                                                                        np.array(class_probs[roof_type]))

        plt.clf()
        for roof_type in utils.ROOF_TYPES:
            plt.plot(recall[roof_type], precision[roof_type],
                    label='Precision-recall curve of class {0} (area = {1:0.2f})'
                                       ''.format(roof_type, average_precision[roof_type]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        path = self.out_path+'AUC.jpg'
        plt.savefig(path)


