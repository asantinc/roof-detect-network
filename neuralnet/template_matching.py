import cv2
from os import listdir
import pdb
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
from collections import defaultdict

import experiment_settings as settings
from reporting import Evaluation, Detections
from timer import Timer
from get_data import DataLoader
from scipy import ndimage



class TemplateMatcher(object):
    def __init__(self, out_path=settings.TEMPLATE_OUT, group_detections=True, contour_detections=True, 
                                    min_neighbors=1, eps=1, in_path=settings.ORIGINAL_TRAINING_PATH):
        self.in_path = in_path

        out_folder = settings.time_stamped('') if group_detections == False else settings.time_stamped('grouped_neigh{0}_eps{1}'.format(min_neighbors, eps))
        self.out_path = out_path+out_folder
        settings.mkdir(self.out_path)

        self.group_detections = group_detections
        self.contour_detections = contour_detections
        self.min_neighbors = min_neighbors
        self.eps = eps

        self.detections = Detections()
        self.detector_names = dict()
        self.detector_names['metal'] = '5 templates'
        self.detector_names['thatch'] = '1 template'
        self.evaluation = Evaluation(method='template', folder_name=out_folder, out_path=self.out_path, detections=self.detections, 
                                                in_path=self.in_path, detector_names=self.detector_names)

        self.templates = dict()

        self.set_metal_templates()
        self.get_thatch_template()
        self.set_thatch_templates()

        self.threshold = 0.5


    def set_metal_templates(self, shape=None):
        templates = list()
        for i in range(5):
            if shape==0 or shape==1 or shape==2:
                template_height, template_width = (40, 60)
            else: # shape == 3 or shape == 4:
                template_height, template_width = (60, 60)

            template = np.zeros((template_height+20, template_width+20))
            template[20:20+template_height, 20:20+template_width] = 255 #white central region 

            if shape == 1 or shape == 3:
                #rotate 45 degrees
                template = ndimage.rotate(template, 45)

            if shape == 2 or shape == 4:
                #rotate 90 degrees
                template = ndimage.rotate(template, 90)

            template_name = 'template{0}.jpg'.format(shape)
            cv2.imwrite(template_name, template)
            template = cv2.imread(template_name, cv2.IMREAD_GRAYSCALE)
            templates.append(template)
        self.templates['metal'] = templates


    def get_thatch_template(self):
        #get some thatch roof
        roof = None
        for img_name in listdir(self.in_path):
            if img_name.endswith('.jpg'):
                roofs = DataLoader().get_roofs(self.in_path+img_name[:-3]+'xml', '' )
                for r in roofs:
                    if r.roof_type == 'thatch':
                        roof = r
                        break
                if roof is not None:
                    break
        #extract patch
        img = cv2.imread(self.in_path+img_name)
        template = img[roof.ymin:roof.ymin+roof.height, roof.xmin:roof.xmin+roof.width]
        img = cv2.imwrite('thatch_template.jpg', template)


    def set_thatch_templates(self):
        template = cv2.imread('thatch_template.jpg', cv2.IMREAD_GRAYSCALE)
        self.templates['thatch'] = [template]


    def detect_score_roofs(self):
        for img_name in listdir(self.in_path):
            if img_name.endswith('.jpg') == False:
                continue
            detections_all = defaultdict(list)
            img_rgb = cv2.imread('{0}{1}'.format(self.in_path, img_name), flags=cv2.IMREAD_COLOR)
            img_gray = cv2.imread('{0}{1}'.format(self.in_path, img_name), cv2.IMREAD_GRAYSCALE)
            img_gray = cv2.equalizeHist(img_gray)
            print '-------------------- Matching.....{0} ------------------------ '.format(img_name)
            group_detected_roofs = dict()
            for roof_type, templates in self.templates.iteritems():
                with Timer() as t:
                    #MATCHING
                    for template in templates:
                        #get detections for each template, keep those over a threshold
                        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                        detections = np.where( res >= self.threshold )
                        w, h = template.shape[::-1]
                        boxes = [(pt[0], pt[1], w, h) for pt in zip(*detections[::-1])]
                        if len(boxes)>0:
                            detections_all[roof_type].extend(boxes)
                
                    print 'Done matching {0}'.format(roof_type)

                    #GROUPING
                    if self.group_detections and self.contour_detections==False:
                        print 'Grouping....'
                        detections_all[roof_type], weights = cv2.groupRectangles(np.array(detections_all[roof_type]).tolist(), self.min_neighbors, self.eps)
                        print 'Done grouping'
                    elif self.contour_detections==True:
                       #do contour detection instead of grouprects 
                        detections_all[roof_type] = self.evaluation.get_bounding_rects(img_name=img_name, rows=img_gray.shape[0], 
                                                        cols=img_gray.shape[1], detections=detections_all[roof_type])
                    print '{0} detections: {1}'.format(roof_type, len(detections_all[roof_type]))

            self.detections.total_time += t.secs
            print 'Time {0}'.format(t.secs)
            self.detections.set_detections(img_name=img_name, detection_list=detections_all)

            self.evaluation.score_img(img_name)
        self.evaluation.print_report()
        self.evaluation.pickle_detections()
        open(self.out_path+'DONE', 'w').close() 


    def detect_roofs(self):
        for img_name in listdir(self.in_path):
            if img_name.endswith('.jpg') == False:
                continue
            detections_all = defaultdict(list)
            img_rgb = cv2.imread('{0}{1}'.format(self.in_path, img_name), flags=cv2.IMREAD_COLOR)
            img_gray = cv2.imread('{0}{1}'.format(self.in_path, img_name), cv2.IMREAD_GRAYSCALE)
            
            print '-------------------- Matching.....{0} ------------------------ '.format(img_name)
            group_detected_roofs = dict()
            for roof_type, templates in self.templates.iteritems():
                with Timer() as t:
                    #MATCHING
                    for template in templates:
                        #get detections for each template, keep those over a threshold
                        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                        detections = np.where( res >= self.threshold )
                        w, h = template.shape[::-1]
                        boxes = [(pt[0], pt[1], w, h) for pt in zip(*detections[::-1])]
                        if len(boxes)>0:
                            detections_all[roof_type].extend(boxes)
                
                    print 'Done matching {0}'.format(roof_type)

                    #GROUPING
                    if self.group_detections:
                        print 'Grouping....'
                        detections_all[roof_type], weights = cv2.groupRectangles(np.array(detections_all[roof_type]).tolist(), self.min_neighbors, self.eps)
                        print 'Done grouping'
                    print '{0} detections: {1}'.format(roof_type, len(detections_all[roof_type]))

            self.detections.total_time += t.secs
            print 'Time {0}'.format(t.secs)
            self.detections.set_detections(img_name=img_name, detection_list=detections_all)
        


    def hist_curve(im):
        h = np.zeros((300,256,3))
        if len(im.shape) == 2:
            color = [(255,255,255)]
        elif im.shape[2] == 3:
            color = [ (255,0,0),(0,255,0),(0,0,255) ]
        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
            cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
            hist=np.int32(np.around(hist_item))
            pts = np.int32(np.column_stack((bins,hist)))
            cv2.polylines(h,[pts],False,col)
        y=np.flipud(h)
        return y


    def hist_lines(im):
        h = np.zeros((300,256,3))
        if len(im.shape)!=2:
            print "hist_lines applicable only for grayscale images"
            #print "so converting image to grayscale for representation"
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        for x,y in enumerate(hist):
            cv2.line(h,(x,0),(x,y),(255,255,255))
        y = np.flipud(h)
        return y


if __name__ == '__main__':
    matcher = TemplateMatcher(group_detections=True, contour_detections=True )
    matcher.detect_score_roofs()


