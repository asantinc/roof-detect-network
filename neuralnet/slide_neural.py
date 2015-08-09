import utils
import math
import os
import cv2
import pdb
from timer import Timer
from reporting import Evaluation, Detections
from collections import defaultdict

class SlidingWindowNeural(object):
    def __init__(self, in_path, out_path, scale=1.5, minSize=(200,200), windowSize=(40,40), stepSize=15):
        self.in_path = in_path
        out_path = out_path
        self.scale = scale
        self.minSize = minSize
        self.windowSize = windowSize
        self.stepSize = stepSize
        self.total_window_num = 0

    def get_windows(self, img_name):
        try:
            image = cv2.imread(self.in_path+img_name)
        except IOError:
            print 'Could not open file'
            sys.exit(-1)

        window_num = 0
        polygons_metal = list()
        polygons_thatch = list()
        rects_metal = list()
        rects_thatch = list()

        #loop through pyramid
        for level, resized in enumerate(utils.pyramid(image, scale=self.scale, minSize=self.minSize)):
            for (x, y, window) in utils.sliding_window(resized, stepSize=self.stepSize, windowSize=self.windowSize):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != self.windowSize[0] or window.shape[1] != self.windowSize[1]:
                    continue
                window_num += 1

                #save the correctly translated coordinates of this window
                polygon = self.get_translated_coords(x, y, level)
                polygons_metal.append(polygon)
                rects_metal.append((x,y,self.windowSize[1], self.windowSize[0]))
                if level == 0: #for thatch we only get polygons at the smallest scale
                    polygons_thatch.append(polygon)
                    rects_thatch.append((x,y,self.windowSize[1], self.windowSize[0]))
        self.total_window_num += window_num
        rects = {'thatch': rects_thatch, 'metal': rects_metal}
        polygons = {'thatch': polygons_thatch, 'metal': polygons_metal}
        return polygons, rects


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

    in_path = utils.get_path(in_or_out=utils.IN, data_fold=data_fold)
    slider = SlidingWindowNeural(in_path=in_path, scale=scale, minSize=minSize, stepSize=stepSize)
    img_names = [img_name for img_name in os.listdir(in_path) if img_name.endswith('.jpg')]
    all_coordinates = dict()

    with Timer() as t:
        for img_name in img_names:
            all_coordinates[img_name] = dict()
            polygons, _ = slider.get_windows(img_name)
            all_coordinates[img_name] = polygons
    print t.secs
    print slider.total_window_num

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

