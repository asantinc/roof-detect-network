import utils
import math
import os
import cv2
import pdb
from timer import Timer
from reporting import Evaluation, Detections
from collections import defaultdict
from get_data import Rectangle

DEBUG = False 


class SlidingWindowNeural(object):
    def __init__(self,  data_fold=None, full_dataset=False, out_path=None, in_path=None, output_patches=False, scale=1.5, minSize=(200,200), windowSize=(40,40), stepSize=15):
        self.scale = scale
        self.minSize = minSize
        self.windowSize = windowSize
        self.output_patches = output_patches 
 
        self.stepSize = stepSize if self.output_patches == False else 30
        self.total_window_num = 0
        if data_fold is None:
            self.data_fold = utils.TRAINING if self.output_patches or full_dataset else utils.VALIDATION
        else:
            self.data_fold = data_fold

        self.in_path = in_path if in_path is not None else utils.get_path(in_or_out=utils.IN, data_fold=self.data_fold, full_dataset=full_dataset)
        self.img_names = [img_name for img_name in os.listdir(self.in_path) if img_name.endswith('.jpg')]
        self.img_names = self.img_names[:20] if DEBUG else self.img_names

        self.detections = Detections()
        folder_name = 'scale{}_minSize{}-{}_windowSize{}-{}_stepSize{}_dividedSizes/'.format(self.scale, self.minSize[0], self.minSize[1], 
                                                self.windowSize[0], self.windowSize[1], self.stepSize) 

        self.out_path = out_path if out_path is not None else '{}'.format(utils.get_path(full_dataset=True, in_or_out=utils.OUT, slide=True, data_fold=self.data_fold, out_folder_name=folder_name))

        self.evaluation = Evaluation(full_dataset=full_dataset, method='slide',folder_name=folder_name, save_imgs=False, out_path=self.out_path, 
                            detections=self.detections, in_path=self.in_path)



    def get_windows_in_folder(self, folder=None):
        folder = folder if folder is not None else self.in_path
        self.all_coordinates = dict()
        with Timer() as t:
            for i, img_name in enumerate(self.img_names):
                print 'Getting windows for image: {}/{}'.format(i+1, len(self.img_names))
                self.all_coordinates[img_name] = dict()
                polygons, rects = self.get_windows(img_name)
                self.all_coordinates[img_name] = rects 
        print t.secs
        self.detections.total_time = t.secs
        print self.total_window_num


    def is_small_image(self, image):
        w, h = image.shape[:2]
        return (w<2000 and h<1000)


    def get_windows(self, img_name):
        try:
            image = cv2.imread(self.in_path+img_name)
        except IOError:
            print 'Could not open file'
            sys.exit(-1)
        if self.is_small_image(image):
            return self.detect(img_name, image)
        else:
            stepSize = 30 if self.output_patches == False else 15
            return self.detect(img_name, image, stepSize=stepSize, windowSize=(40,40), scale=1.5, minSize=(200,200))             


    def detect(self, img_name, image, stepSize=None, windowSize=None, scale=None, minSize=None):
        windowSize = windowSize if windowSize is not None else self.windowSize
        stepSize = stepSize if stepSize is not None else self.stepSize
        scale = scale if scale is not None else self.scale
        minSize = minSize if minSize is not None else self.minSize

        window_num = 0
        polygons_metal = list()
        polygons_thatch = list()
        rects_metal = list()
        rects_thatch = list()

        #loop through pyramid

        for level, resized in enumerate(utils.pyramid(image, scale=scale, minSize=minSize)):
            for (x, y, window) in utils.sliding_window(resized, stepSize=stepSize, windowSize=windowSize):
                
                #self.debug_scaling(image, img_name, resized, x, y, level):

                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
                    continue
                window_num += 1

                #save the correctly translated coordinates of this window
                polygon, rectangle = self.get_translated_coords(x, y, level, scale, windowSize)
                polygons_metal.append(polygon)
                rects_metal.append(rectangle)
                
                polygons_thatch.append(polygon)
                rects_thatch.append(rectangle)
        self.total_window_num += window_num
        rects = {'thatch': rects_thatch, 'metal': rects_metal}
        polygons = {'thatch': polygons_thatch, 'metal': polygons_metal}
        return polygons, rects


    def get_translated_coords(self, x, y, pyramid_level, scale, windowSize):
        scale_factor = math.pow(scale, pyramid_level)
        x = x*scale_factor
        y = y*scale_factor
        w = int(scale_factor*windowSize[1]) #int(scale_factor*self.windowSize[1])
        h = int(scale_factor*windowSize[0]) #int(scale_factor*self.windowSize[0])
        rect = Rectangle(int(x), int(y), int(x+w), int(y+h))
        return utils.convert_rect_to_polygon(rect), rect


    def debug_scaling(self, image, img_name, x, y, level):
        '''Not working
        '''
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + self.windowSize[1], y + self.windowSize[0]), (0, 255, 0), 2)

        clone = image.copy() 
        scale_factor = math.pow(self.scale, level)
        x = (x*scale_factor)
        y = (y*scale_factor)
        w = scale_factor*self.windowSize[1]
        h = scale_factor*self.windowSize[0]
        start = (int(x), int(y))
        end = (int(x + w), int(y + h) )

        cv2.rectangle(clone, start, end, (0, 255, 0), 2)
        b,g,r = cv2.split(clone)
        img2 = cv2.merge([r,g,b])
        plt.subplots(2)
        subplot(2)
        plt.imshow(img2)
        plt.show()

    def run_evaluation(self):
        #EVALUATION
        for i, img_name in enumerate(self.img_names):
            print 'Evaluating image {}/{}'.format(i+1, len(self.img_names))
            #set the detections
            self.detections.set_detections(roof_type='thatch', detection_list=self.all_coordinates[img_name]['thatch'], img_name=img_name)
            self.detections.set_detections(roof_type='metal', detection_list=self.all_coordinates[img_name]['metal'], img_name=img_name)
            #score the image
            self.evaluation.score_img(img_name=img_name, img_shape=(1200,2000), fast_scoring=True)
        self.evaluation.print_report()


def main():
    full_dataset = True
    output_patches = True 
    stepSize = 50 if output_patches else 4
    scale = 1.3
    minSize = (50,50)
    windowSize = (15,15)
    data_fold = None #utils.VALIDATION
    slider = SlidingWindowNeural(data_fold=data_fold, windowSize=windowSize, full_dataset=True, 
                                output_patches=output_patches, stepSize=stepSize, scale=scale, minSize=minSize)
    slider.get_windows_in_folder()
    slider.run_evaluation()
    if output_patches:
        slider.evaluation.save_training_TP_FP_using_voc(rects=True, img_names=slider.img_names)


if __name__ == '__main__':
    main()


