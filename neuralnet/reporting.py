from collections import defaultdict

from viola_detector import Detections
from get_data import Roof, DataLoader #for get_roofs


class Detections(object):
    def __init__(self, roof_types=None):
        self.detections = dict()
        self.total_detection_num = 0
        self.total_time = 0
        self.imgs = set()
        self.roof_types = ['metal', 'thatch'] is roof_types is None else roof_types
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




class Evaluation(object):
    def __init__(self, out_folder=None, detections=None, in_path=None, img_names=None, detector_names=None):
        self.VOC_threshold = 0.5 #threshold to assign a detection as a true positive
        self.scores = defaultdict()

        self.detections = detections    #detection class
        self.in_path = in_path          #the path from which the images are taken
        self.img_names = img_names
        self.roof_types = ['metal', 'thatch'] if roof_types is None else roof_types

        self.roofs = dict()
        for img_path in self.img_names:
            self.roofs[img_path] = Loader().get_roofs(img_path[:-3]+'.xml', img_path)
            #self.overlap_roof_with_detections[img_name] = defaultdict()

        self.out_folder = out_folder
        if detector_names is not None:
            self.init_report(detector_names)


    def init_report(self, detectors):
        self.report_file = self.output_folder+'report.txt'
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


    @staticmethod
    def sum_mask(array):
        '''Sum all ones in a binary array
        '''
        return np.sum(np.sum(array, axis=0), axis=0)


 
    def print_report(self, write_to_file=True):
        log_to_file = list() 
        log_to_file.append('Detections:\t\t{0}'.format(self.detections.total_detection_num))
        log_to_file.append('Time:\t\t{0}'.format(self.detections.total_time))

        for roof_type in ['metal', 'thatch']:
            if self.detector_names[roof_type] == []:
                continue
            # Need to get True positives, True negatives, Recall, Precision and F1
            log_to_file.append('{0}'.format(roof_type))
            log_to_file.append('\t\tTotal roofs\t\tDetections\t\tRecall\t\tPrecision\t\tF1 score\t\t')

        self.scores['total_true_pos']
        self.scores['total_false_pos']
        self.scores['total_detections_pos']

            
        log_to_file.append('\n\n\n')

        with open(self.out_folder+'report.txt', 'a') as report:
            report.write(log_to_file)

        #pickle the evaluation
        with open(self.output_folder+'evaluation.pickle', 'wb') as f:
            pickle.dump(evaluation, f)


    def score_img(self, rows=1200, cols=2000, img_name):
        '''Find best overlap between each roof in an img and the detections,
        according the VOC score
        '''
        voc_scores = list()
        true_pos = 0
        false_pos = 0
        for roof in self.roofs[img_name]:
            best_voc_score = -1
            roof_area = roof.width*roof.height
                    
            roof_mask = np.zeros((rows, cols)) 
            for (x,y,w,h) in self.detections[img_name]:                           #for each patch found
                roof_mask[roof.ymin:roof.ymin+roof.height, roof.xmin:roof.xmin+roof.width] = 1   #the roof
                roof_mask[y:y+h, x:x+w] = 0        #detection
                roof_missed = Evaluation.sum_mask(roof_mask)
                
                intersection_area = roof_area-roof_missed
                detection_area = w*h
                union_area = (roof_area+(detection_area) )-intersection_area
                voc_score = float(intersection_area)/union_area

                if voc_score > best_voc_score:
                    best_voc_score = voc_score
                    x_true, y_true, w_true, h_true = x,y,w,h

            voc_scores.append(best_voc_score)
            if (best_voc_score >= self.VOC_threshold):
                true_pos += 1
            else:
                false_pos += 1

        recall = true_pos / len(self.roofs[img_name])
        precision = true_pos / len(self.detections[img_name])
        self.scores[img_name]['precision'] = precision
        self.scores[img_name]['recall'] = recall
        self.scores[img_name]['voc_scores'] = voc_scores

        self.scores['total_true_pos'] += true_pos
        self.scores['total_false_pos'] += false_neg
        self.scores['total_detections_pos'] += len(self.detections[img_name])
        self.scores['total_roofs'] += len(self.roofs[img_name])

   @staticmethod
    def get_patch_mask(img_name=None, detections=None, rows=1200, cols=2000):
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

   @staticmethod
    def get_detection_contours(patch_path, img_name):
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


    def mark_false_true_detections_and_roofs(self, img_name, roof_list, false_pos, true_pos):
        #get the img
        try:
            img = cv2.imread(self.in_path+img_name, flags=cv2.IMREAD_COLOR)
        except:
            print 'Cannot open {0}'.format(self.in_path+img_name)

        img = self.mark_roofs_on_img(img, img_name, roof_list, (255,0,0))
        img = self.mark_roofs_on_img(img, img_name, false_pos, (0,255,0))
        img = self.mark_roofs_on_img(img, img_name, true_pos, (255,255,255))

        #save image 
        cv2.imwrite(self.out_path+img_name+'_FP_TP.jpg', img) 

    @staticmethod
    def mark_roofs_on_img(self, img, img_name, roof_list, color=(0,255,0)):
        for roof in roof_list:
            if type(roof) is Roof:
                x, y, w, h = roof.xmin, roof.ymin, roof.width, roof.height
            else:
                x, y, w, h = roof
            cv2.rectangle(img,(x,y),(x+w, y+h),color,2)
        return img


    @staticmethod
    def save_detection_img(img_path, img_name, roof_list):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.mark_detections_on_img(img, img_name)
        img = self.mark_roofs_on_img(img, img_name, roof_list)
        cv2.imwrite(self.out_folder+'_'+img_name, img)

    
