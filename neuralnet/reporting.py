from collections import defaultdict

from viola_detector import Detections
from get_data import Roof, DataLoader #for get_roofs

class Evaluation(object):
    def __init__(self, out_folder=None, detections=None, img_name=None  in_path=None, roofs):
        self.VOC_threshold = 0.5 #threshold to assign a detection as a true positive
        self.out_folder = out_folder

        self.roof_list = roof_list
        self.detections = detections #detection class
        self.in_path = in_path #the path from which the images are taken
        self.roof_types = roof_types=['metal', 'thatch'] if roof_types is None else roof_types

       
        self.roofs = dict()
        for img_path in img_list:
            self.roofs[img_path] = Loader().get_roofs(img_path[:-3]+'.xml', img_path)
            self.overlap_roof_with_detections[img_name] = defaultdict()

        self.init_stats()
        self.set_roof_overlap()


    def init_stats(self)
        self.total_area_percentage_covered = defaultdict()
        self.total_area = defaultdict(int)
        self.fully_classified = defaultdict()
        self.mostly_classified = defaultdict()
        self.partially_classified = defaultdict()
        self.missed = defaultdict(int)
        self.near_miss = defaultdict(int)
        self.total_possible_area = defaultdict(int)

    def set_roof_overlap(self):
        for img_name in self.img_names:
            for roof in self.roof_list[img_name]:
                detections=self.detections.get_img_detections_any_type(img_name)
                self.overlap_roof_with_detections[img_name][roof.roof_type].append(Evaluation.max_overlap_single_patch(roof=roof,detections=detections))


    @staticmethod
    def check_overlap_total(self, roof, patch_mask, img_rows, img_cols):
        '''Check overlap between a single roof and any detection
        Not a valid metric according to VOC detection metrics
        '''
        roof_area = roof.width*roof.height
        curr_roof = np.zeros((img_rows, img_cols))
        curr_roof[roof.ymin:roof.ymin+self.height,roof.xmin:roof.xmin+roof.width] = 1
        curr_roof[patch_mask] = 0 
        
        roof_area_found = roof_area-Evaluation.sum_mask(curr_roof)
       
        percent_found = (float(roof_area_found)/roof_area)
        
        #print 'Percent of current roof found: {0}'.format(percent_found)
        return percent_found


    @staticmethod
    def sum_mask(array):
        '''Sum all ones in a binary array
        '''
        return np.sum(np.sum(array, axis=0), axis=0)


    @staticmethod
    def max_VOC_overlap_single_patch(self, rows=1200, cols=2000, detections=None, roof=None):
        '''Return maximum percentage overlap between this roof and a single patch in a set of candidate patches from the same image
        '''
        roof_area = min_miss = roof.width*roof.height
                
        roof_mask = np.zeros((rows, cols)) 
        for (x,y,w,h) in detections:                           #for each patch found
            roof_mask[roof.ymin:roof.ymin+roof.height, roof.xmin:roof.xmin+roof.width] = 1   #the roof
            roof_mask[y:y+h, x:x+w] = 0        #detection
            curr_miss = Evaluation.sum_mask(roof_mask)
            
            #save the coverage percentage
            if curr_miss == 0:                       #we have found the roof
                return 1.0
            elif curr_miss < min_miss:               #keep track of best match
                #add it to the true positives
                min_miss = curr_miss                
                x_true, y_true, w_true, h_true = x,y,w,h
                
        intersection_area = roof_area-curr_miss
        detection_area = w_true*h_true
        union_area = (roof_area+(detection_area) )-intersection_area

        voc_measure = float(intersection_area)/union_area
        #percent_found = (roof_area-min_miss)*(1.)/roof_area

        return voc_measure


   def set_scores(self): 
        '''
        Write to file:
        - the number of roofs detected per file.
        - the false positives and the false negative rates of the joint detector
        - the average time taken for the detection
        '''
        raise ValueError('What is the purpose of this?')
        for roof_type in self.roof_types:
            for img_name in self.overlap_roof_with_detections.keys():
                score_list =  self.overlap_roof_with_detections[img_name][roof_type]

                for roof_score in score_list:
                    if roof_score >= settings.FULLY_CLASSIFIED:
                        fully_classified[roof_type] += 1
                    elif roof_score>= settings.MOSTLY_CLASSIFIED:
                        mostly_classified[roof_type] += 1
                    elif roof_score >= settings.PARTIALLY_CLASSIFIED:
                        partially_classified[roof_type] += 1
                    elif roof_score >= settings.NEAR_MISS:
                        near_miss[roof_type] += 1
                    else:
                        missed[roof_type]+= 1
        
        #dictionary containing stats that we can pickle and unpickle for comparisons between runs
        evaluation = dict() 
        evaluation['path'] = self.output_folder


    def print_report(self, write_to_file=True):
        log_to_file = list() 

        #Print total detections
        detection_num = self.detections.total_detection_num
        evaluation['timing'] = self.detections.total_time

        log_to_file.append('Detections:\t\t{0}'.format(detection_num))
        log_to_file.append('Time:\t\t{0}'.format(self.detections.total_time))


        for roof_type in ['metal', 'thatch']:
            if self.detector_names[roof_type] == []:
                evaluation[roof_type] = False
                continue
            # Need to get True positives, True negatives, Recall, Precision and F1
            log_to_file.append('{0}'.format(roof_type))
            log_to_file.append('\t\t>Recall\t\t>Precision\t\t>F1 score\t\t')

            evaluation[roof_type] = dict() #dict for actual metrics 
            evaluation[roof_type] = 

            
        log_to_file.append('\n\n\n')

        with open(self.out_folder+'report.txt', 'a') as report:
            report.write(log_to_file)

        #pickle the evaluation
        with open(self.output_folder+'evaluation.pickle', 'wb') as f:
            pickle.dump(evaluation, f)



    def find_true_false_positives(self, img_rows, img_cols, img_name, roof_type, threshold=settings.PARTIALLY_CLASSIFIED):
        '''Divide detections between true positives and false positives depending on the percentage of a roof they cover
        '''
        detections = self.detections.get_img_detections_specific_type(roof_type, img_name)
        print 'Detect num: '+str(len(detections))+'\n'
        true_pos = list()
        false_pos_logical = np.empty(len(detections), dtype=bool)

        other_roof_type = 'metal' if roof_type == 'thatch' else 'thatcht'
        other_roofs = [(r.xmin, r.ymin, r.width, r.height) for r in roof_list[img_name] if r.roof_type==other_roof_type]
        other_roofs_type_mask, percent_covered = self.get_patch_mask(detections=other_roofs)
        other_roofs_type_sum = Roof.sum_mask(other_roofs_type_mask)  

        #for every roof, check if any detection covers it enough
        #
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
        self.true_positives = true_pos
        self.false_positives = false_pos 


    def get_patch_mask(self, img_name=None, detections=None, rows=1200, cols=2000):
        '''Mark the area of a detection as True, everything else as False
        '''
        if detections == None:#we either have a list of detections or an img_name
            assert img_name is not None
        detections = self.detections.get_img_detections_any_type(img_name) if detections is None else detections
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
        if self.save_imgs:
            cv2.imwrite(bounding_path, output_bounds)
        return contours


