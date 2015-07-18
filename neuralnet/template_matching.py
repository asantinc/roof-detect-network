import cv2
from os import listdir
import pdb
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt

import experiment_settings as settings
from timer import Timer
from get_data import DataLoader
from scipy import ndimage


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

# def get_template(shape=None, size=None):
#     template = np.zeros((size[0]+20, size[1]+20))
#     print template.shape
#     template[20:20+size[0], 20:20+size[1]] = 255 #white central region 
#     return np.array(template)


class ScoredImage(object):
    def __init__(self, img_path=None, template=None):
        self.img_path = img_path
        self.scores = dict()
        self.template = template
        self.step_size = (self.template.shape[0]/2, self.template.shape[1]/2)

        try:        
            self.img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.gray = cv2.equalizeHist(self.gray)
        except IOError:
            print 'Cannot open '+img_path

    def show_img_histogram(self):
        lines = hist_lines(self.gray)
        cv2.imshow('histogram',lines)
        cv2.imshow('NORMAL', self.gray)
        lines_eq = hist_lines(cv2.equalizeHist(self.gray))
        cv2.imshow('EQUALIZED', cv2.equalizeHist(self.gray))
        cv2.imshow('histogram Equalize',lines_eq)


    def threshold_img(self, threshold=99):
        #only mark with ones the regions where the score is above some threhold
        #coords = [coord for coord, score in self.scores.iteritems() if score > threshold]

        #for coord in coords:
        #    #draw rectangles around the original image
        #    cv2.rectangle(self.img, (coord[0],coord[1]),(self.template.shape[0]+coord[0],
        #                                        coord[1]+self.template.shape[1]), (0,0,255),5)

        percentile = np.percentile(self.scores.values(), threshold)
        percentile *= 255.0/self.img_being_scored.max() #normalize
        normalized_scored_img = self.img_being_scored*255.0/self.img_being_scored.max()

        cv2.imwrite('thres'+str(threshold)+'_'+self.img_out_name, 255*(normalized_scored_img > percentile))
        


    def score(self, test_name=None):
        self.img_out_name = test_name
        self.img_being_scored = np.zeros((1200,2000))
        h, w = self.gray.shape
        vert_patches = ((h - self.template.shape[0]) // self.step_size[0]) + 1  #vert_patches = h/settings.PATCH_H
#        pdb.set_trace()
        print 'Vertical patches {0}'.format(vert_patches)

        #get patches along roof height
        for vertical in range(vert_patches):
            y_pos = (vertical*self.step_size[0])
            self.score_horizontal_patches(y_pos=y_pos)

        #get patches from the last row also
        if (h % self.template.shape[0]>0) and (h > step_size[0]):
            leftover = h-(vert_patches*self.step_size[0])
            y_pos = y_pos-leftover
            self.score_horizontal_patches(y_pos=y_pos)
        
        cv2.imwrite('test_slide.jpg', self.gray)

        self.norm_img_being_scored = self.img_being_scored*255.0/self.img_being_scored.max()
        cv2.imwrite(test_name, self.norm_img_being_scored)





    def score_horizontal_patches(self,y_pos=-1):
        '''Get patches along the width of a patch for a given y_pos (i.e. a given height in the image)
        '''
        h,w = self.gray.shape
        hor_patches = ((w - self.template.shape[1]) // self.step_size[1]) + 1  #hor_patches = w/settings.PATCH_W
        print 'Horizontal patches {0}'.format(hor_patches)
#        pdb.set_trace()
        for i, horizontal in enumerate(range(hor_patches)):
            #get cropped patch
            x_pos = (horizontal*self.step_size[1])
            curr_patch = self.gray[y_pos:y_pos+self.template.shape[0], x_pos:x_pos+self.template.shape[1]]

            diff = np.absolute(curr_patch-self.template)
            score = (diff.sum(axis=0)).sum(axis=0)
            self.img_being_scored[y_pos:y_pos+self.template.shape[0], x_pos:x_pos+self.template.shape[1]] += score
            print score
            coords = (y_pos, x_pos)
            
            self.scores[coords] = score

    def get_scores(self):
        return self.scores.values()


    # in_path = '../data/inhabited/'
    # template_height =  40
    # template_width = 60

    # template = get_template(size=(template_height, template_width))
    # #for img_name in listdir(in_path):
    # for img_name in ['0001.jpg']:
    #     if img_name.endswith('.jpg'):
    #         with Timer() as t:
    #             img_path = in_path+img_name
    #             test_name = '{2}_w{0}_h{1}.jpg'.format(template_width, template_height, img_name)
    #             scored_img = ScoredImage(img_path=img_path, template=template)
    #             #scored_img.show_img_histogram()

    #             scored_img.score(test_name=test_name)
    #             scores = scored_img.get_scores()

    #             for i in [90,91,92,93,94,95,96,97,98,99]:
    #                 scored_img.threshold_img(threshold=i)
    #         print t.secs
    #     break


def get_template(shape=None):
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
    return template
    

if __name__ == '__main__':
    in_path = settings.ORIGINAL_TRAINING_PATH
    loader = DataLoader()
    in_path = '../data/inhabited/'
    for img_name in listdir(in_path):
        if img_name.endswith('.xml'):
            continue
        img_rgb = cv2.imread('../data/inhabited/{0}'.format(img_name), flags=cv2.IMREAD_COLOR)
        img_gray = cv2.imread('../data/inhabited/{0}'.format(img_name), cv2.IMREAD_GRAYSCALE)
        for temp in range(5):
            template = get_template(shape=temp)
            w, h = template.shape[::-1]


            print 'Matching.....{0}'.format(img_name)
            res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
            print 'Done matching'
            threshold = 0.5
            #if temp == 0:
            detections = np.where( res >= threshold)
            for pt in zip(*detections[::-1]):
                color = (0,0,0)
                color = (255,255,255) if temp == 0 else color
                color = (0,255,255) if temp == 1 else color
                color = (255,0,255) if temp == 2 else color
                color = (255,255,0) if temp == 3 else color
                color = (0,0,255) if temp == 4 else color
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), color, 2)

        roofs = loader.get_roofs(in_path+img_name[:-3]+'xml', img_name)
        for roof in roofs:
            cv2.rectangle(img_rgb, (roof.xmin, roof.ymin), (roof.xmin+roof.width, roof.ymin+roof.height), (255,0,0), 2)
        cv2.imwrite('../templating/{1}_thres{0}_multipletemp.jpeg'.format(threshold, img_name),img_rgb)

#1. draw the ground truth
#2. Evaluate it

#Add a square template?
