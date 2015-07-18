import cv2
from os import listdir
import pdb
import numpy as np

import experiment_settings as settings
from timer import Timer

def get_template(shape=None, size=None):
    template = np.zeros((size[0]+20, size[1]+20))
    print template.shape
    template[20:20+size[0], 20:20+size[1]] = 1
    return template


class ScoredImage(object):
    def __init__(self, img_path=None, template=None):
        self.img_path = img_path
        self.scores = dict()
        self.template = template
        self.step_size = (self.template.shape[0]/2, self.template.shape[1]/2)

        try:        
            self.img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.gray = cv2.equalizeHist(gray)
        except IOError:
            print 'Cannot open '+img_path


    def threshold_img(self, threshold=None):
        #only mark with ones the regions where the score is above some threhold
        coords = [coord for coord, score in self.scores.iteritems() if score > threshold]
        for coord in coords:
            #draw rectangles around the original image
            cv2.rectangle(self.img, (coord[0],coord[1]),(self.template.shape[0]+coord[0],
                                                coord[1]+self.template.shape[1]), (0,0,255),5)
        return self.img 


    def score(self):
        self.img_being_scored = np.copy(self.gray)
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
        cv2.imwrite('test_scores.jpg', self.img_being_scored)


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
            print score
            coords = (y_pos, y_pos)
            
            self.scores[coords] = score

    def get_scores(self):
        return self.scores.values()

if __name__ == '__main__':
    in_path = settings.ORIGINAL_TRAINING_PATH
    template_height =  20
    template_width = 40

    template = get_template(size=(template_height, template_width))
    for img_name in listdir(in_path):
        if img_name.endswith('.jpg'):
            with Timer() as t:
                img_path = in_path+img_name
                scored_img = ScoredImage(img_path=img_path, template=template)
                scored_img.score()

                scores = scored_img.get_scores()
                percentile = np.percentile(scores, 25)
                threshold = scored_img.threshold_img(threshold = percentile)
                
                cv2.imwrite('test.jpg', threshold)
            print t.secs
        break
