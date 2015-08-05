from collections import defaultdict
import numpy as np
from os import listdir
import sys
import pdb
import cv2
import pickle
from hashlib import sha1

from get_data import DataLoader #for get_roofs
import utils


class Detections(object):
    def __init__(self, roof_types=None, mergeFalsePos=False):
        self.total_detection_num = defaultdict(int)
        self.false_positive_num =  defaultdict(int)
        self.true_positive_num =  defaultdict(int)
        self.good_detection_num =  defaultdict(int)

        if mergeFalsePos:
            self.bad_detection_num = 0 #we count the metal and thatch false positives together 
        else:
            self.bad_detection_num = defaultdict(int)
        self.roof_num =  defaultdict(int)

        self.detections = defaultdict(list)
        self.true_positives = dict()
        self.false_positives = dict()
        self.good_detections =dict()#all the detections above the VOC threshold
        #index with the roof polygon coordinates
        #stores the best score and detection for each grount true roof
        self.roof_detections_voc = dict() 

        if mergeFalsePos:
            self.bad_detections = defaultdict(list) #there's only one set of bad detections
        else:
            self.bad_detections = dict()
        for roof_type in utils.ROOF_TYPES: # we need this to separate them per image
            self.roof_detections_voc[roof_type] = dict()
            self.detections[roof_type] = dict()
            self.true_positives[roof_type] = defaultdict(list)
            self.false_positives[roof_type] = defaultdict(list)
            self.good_detections[roof_type] = defaultdict(list)
            if mergeFalsePos == False:
                self.bad_detections[roof_type] = defaultdict(list)

        self.total_time = 0
        self.imgs = set()

    def update_best_voc(self,img_name=None, roof_type=None, roof_polygon=None, best_detection=None, score=None):
        roof = sha1(roof_polygon) #we can't use np array as a hash, so we make it a tuple
        if img_name not in self.roof_detections_voc[roof_type]:
            self.roof_detections_voc[roof_type][img_name] = dict()
        #ensure the new score is actually better
        if roof in self.roof_detections_voc[roof_type][img_name]:
            assert self.roof_detections_voc[roof_type][img_name][roof][1] < score
        self.roof_detections_voc[roof_type][img_name][roof] = (best_detection, score)  

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



class Evaluation(object):
    def __init__(self, report_name=None, method=None, 
                        folder_name=None, save_imgs=True, out_path=None, 
                        detections=None, in_path=None, 
                        detector_names=None, 
                        mergeFalsePos=False,
                        separateDetections=True,
                        vocGood=0.1):
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

        self.detections = detections    #detection class

        self.in_path = in_path          #the path from which the images are taken
        self.img_names = [f for f in listdir(self.in_path) if f.endswith('.jpg')]

        #the ground truth roofs for every image and roof type
        self.correct_roofs = dict()
        for roof_type in utils.ROOF_TYPES:
            self.correct_roofs[roof_type] = dict()
        for img_name in self.img_names:
            for roof_type in utils.ROOF_TYPES:
                self.correct_roofs[roof_type][img_name] = DataLoader.get_polygons(roof_type=roof_type, 
                                                            xml_name=img_name[:-3]+'xml' , xml_path=self.in_path)
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

    
    def score_img(self, img_name, img_shape, contours=False):
        '''Find best overlap between each roof in an img and the detections,
        according the VOC score
        '''
        print 'Scoring.....'
        #start with all detections as false pos; as we find matches, we increase the true_pos, decrese the false_pos numbers
        detections = dict() 
        false_pos_logical = dict()
        bad_detection_logical = dict()
        

        detections = dict()
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
            false_pos_logical[roof_type] = np.ones(len(detections[roof_type]), dtype=bool) #[roof_type]
            bad_detection_logical[roof_type] = np.ones(len(detections[roof_type]), dtype=bool) #[roof_type]

            for roof in self.correct_roofs[roof_type][img_name]:
                best_voc_score = -1 
                best_detection = -1 


                for d, detection in enumerate(detections[roof_type]):  # detections[roof_type]):                           #for each patch found
                    #detection_roof_portion is how much of the detection is covered by roof
                    voc_score, detection_roof_portion = self.get_score(contours=work_with_contours, 
                                                        rows=img_shape[0], cols=img_shape[1], roof=roof, detection=detection)
                    if (voc_score > self.VOC_threshold) and (voc_score > best_voc_score):#this may be a true pos
                        best_voc_score = voc_score
                        best_detection = d 

                    #store the best detection regardless of whether it is over 0.5
                    if voc_score > best_voc_score:
                        self.detections.update_best_voc(img_name=img_name, roof_type=roof_type, roof_polygon=roof, best_detection=detection, score=voc_score)  

                    #if self.output_patches:
                    #depending on roof type we consider a different threshold
                    if (voc_score > self.VOC_good_detection_threshold[roof_type] or detection_roof_portion>self.detection_portion_threshold):
                        bad_detection_logical[roof_type][d] = 0 #we already know that this wasn't a bad detection

                if best_detection != -1:
                    false_pos_logical[roof_type][best_detection] = 0 #this detection is not a false positive

        self.update_scores(img_name, detections, false_pos_logical, bad_detection_logical)
        self.save_images(img_name)

       

    def update_scores(self, img_name, detections, false_pos_logical, bad_detection_logical):

        if self.mergeFalsePos: #self.output_patches 
            detects = np.array(detections['metal'].extend(detections['thatch']))
            #a detection is only considered bad if it doesn't match either a metal or a thatch roof
            truly_bad_detections = np.logical_and(bad_detection_logical['metal'], bad_detection_logical['thatch']) 
            bad_d = detects[truly_bad_detections]
            self.detections.update_bad_detections(bad_detections=bad_d, img_name=img_name) 

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


    def print_report(self, write_to_file=True):
        log_to_file = list() 
        print '*************** FINAL REPORT *****************'
        log_to_file.append('Total Detection Time:\t\t{0}'.format(self.detections.total_time))

        for roof_type in utils.ROOF_TYPES:
            log_to_file.append('Roof type: {0}'.format(roof_type))
            log_to_file.append('Total roofs\tDetections\tRecall\tPrecision\tF1 score\t')

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

            log_to_file.append('{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}\n'.format(cur_type_roofs, detection_num, recall, precision,F1)) 
           
        log_to_file.append('\n\n\n')
        log = '\n'.join(log_to_file)
        print log
        with open(self.out_path+'report.txt', 'a') as report:
            report.write(log)
 




    def save_images(self, img_name, fname=''):
        '''Displays the ground truth, along with the true and false positives for a given image
        '''
        try:
            img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
        except IOError:
            print 'Cannot open {0}'.format(self.in_path+img_name)
            sys.exit(-1)
        for roof_type in utils.ROOF_TYPES:
            utils.draw_detections(self.detections.false_positives[roof_type][img_name], img, color=(0, 0, 255))
            utils.draw_detections(self.correct_roofs[roof_type][img_name], img, color=(0, 0, 0))
            utils.draw_detections(self.detections.true_positives[roof_type][img_name], img, color=(0, 255, 0))


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



