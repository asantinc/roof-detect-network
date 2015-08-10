import utils
import math
import os
import cv2
import pdb
from timer import Timer
from reporting import Evaluation, Detections
from collections import defaultdict

DEBUG = False 


class SlidingWindowNeural(object):
    def __init__(self,  out_path=None, in_path=None, output_patches=False, scale=1.5, minSize=(200,200), windowSize=(40,40), stepSize=15):
        self.scale = scale
        self.minSize = minSize
        self.windowSize = windowSize
        self.stepSize = stepSize
        self.total_window_num = 0
        self.data_fold = utils.TRAINING if output_patches else utils.VALIDATION

        self.in_path = in_path if in_path is not None else utils.get_path(in_or_out=utils.IN, data_fold=self.data_fold)
        self.img_names = [img_name for img_name in os.listdir(self.in_path) if img_name.endswith('.jpg')]
        #self.img_names = self.img_names[:2] if DEBUG else self.img_names

        self.detections = Detections()
        folder_name = 'scale{}_minSize{}-{}_stepSize{}/'.format(self.scale, self.minSize[0], self.minSize[1], self.stepSize) 

        self.out_path = out_path if out_path is not None else '{}'.format(utils.get_path(in_or_out=utils.OUT, slide=True, data_fold=self.data_fold, out_folder_name=folder_name))
        self.evaluation = Evaluation(method='slide',folder_name=folder_name, save_imgs=False, out_path=self.out_path, 
                            detections=self.detections, in_path=self.in_path)



    def get_windows_in_folder(self, folder=None):
        folder = folder if folder is not None else self.in_path
        self.all_coordinates = dict()
        with Timer() as t:
            for img_name in self.img_names:
                self.all_coordinates[img_name] = dict()
                polygons, _ = self.get_windows(img_name)
                self.all_coordinates[img_name] = polygons
        print t.secs
        self.detections.total_time = t.secs
        print self.total_window_num



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
        if DEBUG == False:
            rects = {'thatch': rects_thatch, 'metal': rects_metal}
            polygons = {'thatch': polygons_thatch, 'metal': polygons_metal}
        else:
            rects = {'thatch': rects_thatch[:10], 'metal': rects_metal[:10]}
            polygons = {'thatch': polygons_thatch[:10], 'metal': polygons_metal[:10]}

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

    def run_evaluation(self):
        #EVALUATION
        for i, img_name in enumerate(self.img_names):
            print 'Evaluating image {}/{}'.format(i+1, len(self.img_names))
            #set the detections
            self.detections.set_detections(roof_type='thatch', detection_list=self.all_coordinates[img_name]['thatch'], img_name=img_name)
            self.detections.set_detections(roof_type='metal', detection_list=self.all_coordinates[img_name]['metal'], img_name=img_name)
            #score the image
            self.evaluation.score_img(img_name=img_name, img_shape=(1200,2000))
        self.evaluation.print_report()


def main():
    output_patches = True
    stepSize = 30 if output_patches else 15
    slider = SlidingWindowNeural(output_patches=output_patches, stepSize=stepSize)
    slider.get_windows_in_folder()
    slider.run_evaluation()
    if output_patches:
        slider.evaluation.save_training_TP_FP_using_voc(img_names=slider.img_names)


if __name__ == '__main__':
    main()


