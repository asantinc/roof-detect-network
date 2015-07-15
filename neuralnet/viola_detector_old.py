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
from viola_trainer import ViolaTrainer, ViolaDataSetup
from timer import Timer


class DetectorMetrics(object):
    def __init__(self, roof_type=None, name=None):
        self.name = detector_name
        self.roof_type = roof_type
 
        #track the amount of overlap between a roof and detections 
        self.overlap_roof_with_detections = defaultdict(list)


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
        self.roofs_detected = dict()
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
            self.roof_detectors['metal'] = [cv2.CascadeClassifier('../viola_jones/'+path+'/cascade.xml') for path in detector_names['metal']]
            self.roof_detectors['thatch'] = [cv2.CascadeClassifier('../viola_jones/'+path+'/cascade.xml') for path in detector_names['thatch']]



    def detect_roofs(self, img_name=None, img_path=None, reject_levels=1.3, level_weights=5, scale=None):
        if self.neural==False:
            if os.path.isfile(self.output_folder+img_name+'.pickle'):
                #load from file
                with open(self.output_folder+img_name+'.pickle', 'rb') as f:
                    self.roofs_detected[img_name] = pickle.load(f) 

        else:
            img = cv2.imread(img_path+img_name, flags=cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.scale = scale if scale is not None else self.scale

            #get detections in the image
            roof_detections = dict()
            self.total_detection_time = 0
            for roof_type in self.roof_detectors.keys():
                roof_detections[roof_type] = list()
                for i, detector in enumerate(self.roof_detectors[roof_type]):
                    print 'Detecting with detector: '+str(i)
                    with Timer() as t: 
                        detected_roofs = detector.detectMultiScale(gray, scaleFactor=self.scale, minNeighbors=5)
                    self.total_detection_time += t.secs
                    #group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
                    roof_detections[roof_type].append(detected_roofs)

            self.roofs_detected[img_name] = roof_detections
           
            #save detections to file
            if self.neural==False:
                with open(self.output_folder+img_name+'.pickle', 'wb') as f:
                    pickle.dump(self.roofs_detected[img_name], f)
                 

    def detect_evaluate_roofs_folder(self, path=settings.INHABITED_PATH, save_detections=False, reject_levels=1.3, level_weights=5, scale=1.05):
        '''Compare detections to ground truth roofs for set of images in a folder
        '''
        if self.load_pickle_stats == False:
            loader = get_data.DataLoader()
            img_names = get_data.DataLoader.get_img_names_from_path(path=path)
            for i, img_name in enumerate(img_names):
                self.detect_evaluate_roofs(loader, img_name=img_name, path=path, reject_levels=reject_levels, level_weights=level_weights, scale=scale)
             
            #we can store the stats to use them directly
            if self.pickle_dump_stats:
                with open(self.output_folder+'stats.pickle', 'wb') as f:
                    pickle.dump(self.overlap_roof_with_detections, f)
        else:
            with open(self.output_folder+'stats.pickle', 'rb') as f:
                self.overlap_roof_with_detections = pickle.load(f)

        self.print_report()
   

    def detect_evaluate_roofs(self, 
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

        
    def match_roofs_to_detection(self, img_name, roof_list, rows=1200, cols=2000):
        #Compare detections to ground truth
        patch_mask, detection_percent = self.get_patch_mask(img_name)
        #print 'Percent of image covered by detection:{0}'.format(detection_percent)
        patch_location = self.output_folder+img_name+'_mask.jpg'
        if self.save_imgs:
            cv2.imwrite(patch_location, np.array(patch_mask, dtype=int))

        #initialize the detection evaluation metrics for the current image
        self.overlap_roof_with_detections[img_name] = defaultdict()
        for roof_type in ['thatch', 'metal']:
            self.overlap_roof_with_detections[img_name][roof_type] = defaultdict(list)
            for coverage_type in ['any', 'single']: 
                self.overlap_roof_with_detections[img_name][roof_type][coverage_type] = list()

        detection_contours = self.get_detection_contours(patch_location, img_name)
        for roof in roof_list:
            self.overlap_roof_with_detections[img_name][roof.roof_type]['any'].append(roof.check_overlap_total(patch_mask, rows, cols))
            self.overlap_roof_with_detections[img_name][roof.roof_type]['single'].append(roof.max_overlap_single_patch(detections=self.roofs_detected[img_name]))


    def get_neural_training_data(self, roof_type=None, out_path=None, in_path=None, detector_name=None):
        '''
        Run detection to get false positives, true positives from viola jones. Save these patches.
        These results will be used to then train a neural network.
        '''
        self.true_pos = dict()
        self.false_pos = dict()

        loader = get_data.DataLoader()
        img_names = get_data.DataLoader.get_img_names_from_path(path=in_path)

        for num, img_name in enumerate(img_names):
            print 'Processing image: '+str(num)+', '+img_name+'\n' 
            self.detect_roofs(img_name=img_name, img_path=in_path)
            
            #get roofs
            xml_path = in_path+img_name[:-3]+'xml'
            roof_list = loader.get_roofs(xml_path, img_name)

            #get true and false positives
            self.true_pos[img_name], self.false_pos[img_name] = self.find_true_false_positives(1200, 2000, img_name, roof_type, roof_list) 

            #save roof patches to file
            self.save_patches_to_folder(in_path=in_path, img_name=img_name, out_path=out_path+'true_pos/'+detector_name, patches=self.true_pos[img_name])
            self.save_patches_to_folder(in_path=in_path, img_name=img_name, out_path=out_path+'false_pos/'+detector_name, patches=self.false_pos[img_name])
            self.mark_false_true_detections_and_roofs(out_path, in_path, img_name, roof_list, 
                                                            self.false_pos[img_name], self.true_pos[img_name])


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
        detections = [d for detector in self.roofs_detected[img_name][roof_type] for d in detector]
        print 'Detect num: '+str(len(detections))+'\n'
        true_pos = list()
        false_pos_logical = np.empty(len(detections), dtype=bool)

        for d, (x,y,w,h) in enumerate(detector):                           #for each patch found
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
                    false_pos_logical[d] = True
                    #remove the true pos from the false positives
        det = np.array(detections)
        false_pos = det[false_pos_logical]
        return true_pos, false_pos 


    def report_roofs_detected(self, img_name):
        for roof_type in ['metal', 'thatch']:
            for i, detector in enumerate(self.detector_names[roof_type]):
                print 'Detector {0}: \t {1}'.format(detector, len(self.roofs_detected[img_name][roof_type][i]) )



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
            detection_num=0
            for img in self.roofs_detected.keys():
                for detection_list in self.roofs_detected[img_name]:
                    detection_num += len(detection_list)
            evaluation['detection_num']=detection_num
            report.write('Detections:\t\t{0}'.format(detection_num)+'\n')
            #Print timing
            
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
                    pdb.set_trace()
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

    '''
    def print_report(self, img_name=None, detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as report:
                self.thatch_candidates += len(self.overlap_dict[img_name]["thatch"])
                self.metal_candidates += len(self.overlap_dict[img_name]["metal"])
                        
                #report true positives
                true_metal = true_thatch = 0
                for roof_type in self.overlap_dict[img_name].keys():
                    fully_classified, mostly_classified, partially_classified = self.score_detections(roof_type=roof_type, img_name=img_name)

                    self.overlap_all[roof_type]['any'] += sum(self.overlap_roof_with_detections[img_name][roof_type]['any']) 
                    self.overlap_all[roof_type]['single'] += sum(self.overlap_roof_with_detections[img_name][roof_type]['single']) 

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
    '''

    def mark_detections_on_img(self, img, img_name):
        '''Return an image with the detections and the ground truth roofs marked with rectangles
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

def get_detectors():
    detectors = dict()
    combo_f = raw_input('Enter detector combo file number: ')
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
                detector['thatch'].append(line[1].strip())
            else:
                raise ValueError("Unknown detector type {0}".format(line[0]))
    return detectors, combo_f 


if __name__ == '__main__':
    #TESTING THE DETECTORS
    #can have multiple detectors for each type of roof   
    '''
    detectors, combo_f_name = get_detectors()
    testing = raw_input('Type "t" to use testing_new_source as the test set:')
    out_path = '../output/viola/with_testing_set/' if testing == 't' else '../output/viola/with_training_set/'
    in_path = settings.TESTING_NEW_SOURCE if testing else settings.INHABITED_PATH   
    folder_name = 'combo'+combo_f_name+'_'+settings.time_stamped('/')
    viola = ViolaDetector(load_pickle_stats = False, folder_name=folder_name, out_path=out_path, detector_names=detectors, save_imgs=False, old_detector = False)
    viola.detect_evaluate_roofs_folder(path=in_path, save_detections=True, reject_levels=0.5, level_weights=2, scale=1.05)
    '''

    #Getting patches for neural network
    detectors, combo_f_name = get_detectors()
    in_path = settings.INHABITED_PATH  
    out_path = settings.TRAIN_NEURAL_VIOLA_EXTRA
    viola = ViolaDetector(load_pickle_stats = False, detector_names=detectors, save_imgs=True, old_detector = False, neural=True)
    viola.get_neural_training_data(roof_type='metal', out_path=out_path, in_path=in_path, detector_name='combo'+combo_f_name) 


