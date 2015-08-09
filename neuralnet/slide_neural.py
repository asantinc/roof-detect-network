import utils
import math
import os
import cv2
import pdb
from timer import Timer
from reporting import Evaluation, Detections
from collections import defaultdict

class SlidingWindowNeural(object):
    def __init__(self, scale=1.5, minSize=(200,200), windowSize=(40,40), stepSize=15):
        self.scale = scale
        self.minSize = minSize
        self.windowSize = windowSize
        self.stepSize = stepSize

    def slide_classify(self, image, img_name):
        #loop over the image pyramid
        window_num = 0
        coordinates_metal = list()
        coordinates_thatch = list()
        for level, resized in enumerate(utils.pyramid(image, scale=self.scale, minSize=self.minSize)):
            for (x, y, window) in utils.sliding_window(resized, stepSize=self.stepSize, windowSize=self.windowSize):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != self.windowSize[0] or window.shape[1] != self.windowSize[1]:
                    continue
                window_num += 1

                #save the correctly translated coordinates of this window
                polygon = self.get_translated_coords(x, y, level)
                coordinates_metal.append(polygon)
                if level == 0: #for thatch we only get polygons at the smallest scale
                    coordinates_thatch.append(polygon)
        return coordinates_thatch, coordinates_metal, window_num


    def get_translated_coords(self, x, y, pyramid_level):
        if pyramid_level > 0:
            scale_factor = math.pow(self.scale, pyramid_level)
            x = x*scale_factor
            y = y*scale_factor
            w = int(scale_factor*self.windowSize[1])
            h = int(scale_factor*self.windowSize[0])
            rect = (int(x), int(y), w, h)
        else:
            rect = (x, y, self.windowSize[1], self.windowSize[0])
        return utils.convert_rect_to_polygon(rect)


    def debug_scaling(self, image, img_name, resized, x, y, level):
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + self.windowSize[1], y + self.windowSize[0]), (0, 255, 0), 2)
        cv2.imwrite('debug/{}_level{}_S.jpg'.format(img_name[:-4], level), clone)

        clone = image.copy() 
        scale_factor = math.pow(self.scale, level)
        x = (x*scale_factor)
        y = (y*scale_factor)
        w = scale_factor*self.windowSize[1]
        h = scale_factor*self.windowSize[0]
        start = (int(x), int(y))
        end = (int(x + w), int(y + h) )
        cv2.rectangle(clone, start, end, (0, 255, 0), 2)
        cv2.imwrite('debug/{}_level{}_L.jpg'.format(img_name[:-4], level), clone)




if __name__ == '__main__':
    scale = 1.5
    minSize = (200,200)
    stepSize= 15
    output_patches = False
    data_fold = utils.TRAINING if output_patches else utils.VALIDATION

    slider = SlidingWindowNeural(scale=scale, minSize=minSize, stepSize=stepSize)
    in_path = utils.get_path(in_or_out=utils.IN, data_fold=data_fold)
    img_names = [img_name for img_name in os.listdir(in_path) if img_name.endswith('.jpg')]
    total_windows = 0
    all_coordinates = dict()
    with Timer() as t:
        for img_name in img_names:
            all_coordinates[img_name] = dict()
            image = cv2.imread(in_path+img_name)
            coordinates_thatch, coordinates_metal, window_num = slider.slide_classify(image, img_name)
            all_coordinates[img_name]['thatch'] = coordinates_thatch
            all_coordinates[img_name]['metal'] = coordinates_metal
            print img_name, str(window_num)
            total_windows = total_windows+window_num
    print t.secs
    print total_windows

    #EVALUATION
    detections = Detections()
    folder_name = 'scale{}_minSize{}-{}_stepSize{}/'.format(scale, minSize[0], minSize[1], stepSize) 
    out_path = utils.get_path(in_or_out=utils.OUT, slide=True, data_fold=data_fold)
    evaluation = Evaluation(method='slide',folder_name=folder_name, save_imgs=False, out_path=out_path, 
                        detections=detections, in_path=in_path)
    pdb.set_trace()
    for img_name in img_names:
        #set the detections
        detections.set_detections(roof_type='thatch', detection_list=all_coordinates[img_name]['thatch'], img_name=img_name)
        detections.set_detections(roof_type='metal', detection_list=all_coordinates[img_name]['metal'], img_name=img_name)
        #score the image
        evaluation.score_img(img_name=img_name, img_shape=(1200,2000))
    evaluation.print_report()

