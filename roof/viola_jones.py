import os
import subprocess
import pdb

import numpy as np
import cv2
from scipy import misc, ndimage #load images

import get_data
import experiment_settings as settings

MIN_OVERLAP_PERCENT = 0.80


class DataAugmentation(object):
    def transform(self, Xb, yb):
        self.Xb = Xb
        self.random_flip()
        self.random_rotation(10.)
        self.random_crop()
        return self.Xb, yb

    def random_rotation(self, ang, fill_mode="nearest", cval=0.):
        angle = np.random.uniform(-ang, ang)
        self.Xb = scipy.ndimage.interpolation.rotate(self.Xb, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)

    def random_crop(self):
        #Extract 32 by 32 patches from 40 by 40 patches, rotate them randomly
        temp_Xb = np.zeros((self.Xb.shape[0],self.Xb.shape[1], CROP_SIZE, CROP_SIZE))
        margin = IMG_SIZE-CROP_SIZE
        for img in range(self.Xb.shape[0]):
            xmin = np.random.randint(0, margin)
            ymin = np.random.randint(0, margin)
            temp_Xb[img, :,:,:] = self.Xb[img, :, xmin:(xmin+CROP_SIZE), ymin:(ymin+CROP_SIZE)]
        self.Xb = temp_Xb

    def random_flip(self):
        # Flip half of the images in this batch at random:
        bs = self.Xb.shape[0]
        indices_hor = np.random.choice(bs, bs / 2, replace=False)
        indices_vert =  np.random.choice(bs, bs / 2, replace=False)
        self.Xb[indices_hor] = self.Xb[indices_hor, :, :, ::-1]
        self.Xb[indices_vert] = self.Xb[indices_vert, :, ::-1, :]


class Viola(object):

    def __init__(self, num_pos, num_neg, 
            roof_type, 
            detector_paths=None, 
            report_path=None,
            output_folder=None
            ):

        self.roofs_detected = dict()
        self.roof_detectors = list()

        self.detector_paths = detector_paths
        for path in self.detector_paths:
            self.roof_detectors.append(cv2.CascadeClassifier(path))

        #output names
        self.reset_params(num_pos=num_pos, num_neg=num_neg, roof_type=roof_type)

        self.output_folder = output_folder
        self.file_name_default = '''numPos{0}_numNeg{1}_scale{2}_{3}'''.format(num_pos, num_neg, self.scale, roof_type)
        self.report_path = self.output_folder



    def reset_params(self, num_pos=None, num_neg=None, roof_type=None, scale=1.05):
        self.num_pos = num_pos if num_pos is not None else self.num_pos
        self.num_neg = num_neg if num_neg is not None else self.num_neg
        self.roof_type = roof_type if roof_type is not None else self.roof_type
        self.scale = scale
        self.total_roofs = 0
        self.true_positives = 0
        self.all_detections_made = 0  


    @staticmethod
    def setup_negative_samples():
        output = open('../data/bg.txt', 'w')
        for file in os.listdir(settings.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(settings.UNINHABITED_PATH+file+'\n')


    @staticmethod
    def get_dat_string(roof_list, img_path):
        return img_path+'\t'+str(len(roof_list))+'\t'+'\t'.join(roof_list)+'\n'


    @staticmethod
    def setup_positive_samples():
        metal_n = '../data/metal.dat'
        thatch_n = '../data/thatch.dat'

        with open(metal_n, 'w') as metal_f, open(thatch_n, 'w') as thatch_f:
            img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.INHABITED_PATH)
            roof_loader = get_data.DataLoader()

            for img_name in img_names_list:
                xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
                img_path = settings.INHABITED_PATH+img_name
        
                roofs, _, _ = roof_loader.get_roofs(xml_path)
                metal_log = list()
                thatch_log = list()

                for roof in roofs:
                    #append roof characteristics separated by single space
                    roof_info = str(roof.xmin)+' '+str(roof.ymin)+' '+str(roof.width)+' '+str(roof.height)
                    if roof.roof_type == 'metal':
                        metal_log.append(roof_info)
                    elif roof.roof_type == 'thatch':
                        roof_info = str(roof.xmin)+' '+str(roof.ymin)+' '+str(30)+' '+str(30)
                        thatch_log.append(roof_info)

                if len(metal_log)>0:
                    metal_f.write(get_dat_string(metal_log, img_path))
                if len(thatch_log)>0:
                    thatch_f.write(get_dat_string(thatch_log, img_path))


    def detect_roofs(self, img_name, reject_levels=1.3, level_weights=5):
        img = cv2.imread(settings.INHABITED_PATH+img_name, flags=cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        roofs = list()
        for detector in self.roof_detectors:
            detected_roofs = detector.detectMultiScale(
                        gray,
                        scaleFactor=self.scale,
                        minNeighbors=5)
            roofs.append(detected_roofs)
        return roofs


    def report_roofs_detected(self, i, img_name):
        detected_numbers = list()
        for detection_set in self.roofs_detected[img_name]:
            detected_numbers.append(len(detection_set))
        print '*************************** IMAGE:'+str(i)+','+str(img_name)+'***************************'
        print 'Roofs detected: '+str(detected_numbers)+' Total: '+str(sum(detected_numbers))


    def test_viola(self, reject_levels=1.3, level_weights=5, scale=1.05):
        self.reset_params(scale=scale)

        #open report file
        self.report_file = self.report_path+'report_scale'+str(scale)+'.txt'
        open(self.report_file, 'w').close()

        self.overlap_dict = dict()
        loader = get_data.DataLoader()
        img_names = get_data.DataLoader.get_img_names_from_path(path=settings.INHABITED_PATH)

        for i, img_name in enumerate(list(img_names)):
            self.roofs_detected[img_name] = self.detect_roofs(img_name)
            self.report_roofs_detected(i, img_name)

            img_path = settings.INHABITED_PATH+img_name

            try:
                image = misc.imread(img_path)
            except IOError:
                print 'Cannot open '+img_path
            rows, cols, _ = image.shape

            xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
            roof_list, _, _ = loader.get_roofs(xml_path)
            
            #Compare detections to ground truth
            self.overlap_dict[img_name] = list()
            for roof in roof_list:
                if roof.roof_type == self.roof_type:
                    self.overlap_dict[img_name].append(roof.max_overlap_single_patch(rows=rows, cols=cols, 
                                detections=self.roofs_detected[img_name]))
                else:
                    continue
            
            img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            img = self.mark_detections_on_img(img, img_path, img_name)
            img = self.mark_roofs_on_img(img, img_path, img_name, roof_list)
            self.save_image(img, img_name)

            self.print_report(img_name)

        self.print_report(final_stats=True)


    def print_report(self, img_name='', detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as output:
            if not final_stats:
                detections_made = len(self.roofs_detected[img_name])
                self.all_detections_made += detections_made 

                total_roofs = len(self.overlap_dict[img_name])
                found = 0
                for v in self.overlap_dict[img_name]:
                    self.total_roofs += 1
                    if v > 0.20:
                        found += 1
                        self.true_positives += 1
                output.write('Image '+img_name+': '+str(found)+'/'+str(total_roofs)+'\n')

            elif final_stats:
                #Print number of false positives, False negatives
                output.write('******************************* RESULTS ********************************* \n')
                output.write('Precision: \t'+str(self.true_positives)+'/'+str(self.all_detections_made)+'\n')
                output.write('Recall: \t'+str(self.true_positives)+'/'+str(self.total_roofs)+'\n')

                with open(self.report_path, 'a') as out_all:
                    out_all.write('******************************* RESULTS ********************************* \n')
                    out_all.write('Params: \t Scale:'+str(self.scale)+'\t NumPos'+str(self.num_pos)+'\t Num_neg:'+str(self.num_neg)+'\n')
                    out_all.write('Precision: \t'+str(self.true_positives)+'/'+str(self.all_detections_made)+'\n')
                    out_all.write('Recall: \t'+str(self.true_positives)+'/'+str(self.total_roofs)+'\n')
                self.true_positive = 0
                self.total_roofs = 0
                self.all_detections_made = 0


    def mark_detections_on_img(self, img, img_path, img_name):
        ''' Save an image with the detections and the ground truth roofs marked with rectangles
        '''
        for i, cascade in enumerate(self.roofs_detected[img_name]):
            for (x,y,w,h) in cascade:
                if i==0:
                    color=(255,0,0)
                elif i==1:
                    color=(0,0,255)
                elif i==2:
                    color=(255,255,255)
                cv2.rectangle(img,(x,y),(x+w,y+h), color, 2)
        return img


    def mark_roofs_on_img(self, img, img_path, img_name, roof_list):
        for roof in roof_list:
            if roof.roof_type == self.roof_type:
                cv2.rectangle(img,(roof.xmin,roof.ymin),(roof.xmin+roof.width,roof.ymin+roof.height),(0,255,0),2)
        return img


    def save_image(self, img, img_name):        
        cv2.imwrite(self.output_folder+self.file_name_default+'_'+img_name, img)


def get_max_metal():
    ''' Get size of largest metal house
    '''
    loader = get_data.DataLoader()
    img_names = get_data.DataLoader.get_img_names_from_path(path=settings.INHABITED_PATH)

    for i, img_name in enumerate(list(img_names)):
        img_path = settings.INHABITED_PATH+img_name
        try:
            image = misc.imread(img_path)
        except IOError:
            print 'Cannot open '+img_path
        xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
        roof_list, _, _ = loader.get_roofs(xml_path)
        max_h = -1
        max_w = -1
        for roof in roof_list:
            if roof.height > max_h:
                max_h = roof.height
            if roof.width > max_w:
                max_w = roof.width
    print max_h, max_w


def frange(start, end=None, inc=None):
    "A range function, that does accept float increments."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
        
    return L

 
def detect_metal():    #set up the data
    #Viola.setup_negative_samples()
    #Viola.setup_positive_samples()
    #train the cascade

    num_pos = 209
    num_neg = 700
    roof_type = 'metal'
    # detectors_2 = ['../viola_jones/cascade_w12_h25_pos393_neg1500.xml', 
    #             '../viola_jones/cascade_w25_h12_pos393_neg1500.xml', 
    #             '../viola_jones/cascade_simple_1500.xml']
    
    detectors_2 = ['../viola_jones/cascade_w12_h25_pos393_neg700.xml', 
                '../viola_jones/cascade_w25_h12_pos393_neg700.xml', 
                '../viola_jones/cascade_simple_1500.xml']
    output_folder = '../viola_jones/output_fewer_negatives_test/'
    
    viola = Viola(num_pos, num_neg, roof_type, detector_paths=detectors_2, output_folder=output_folder)
    viola.test_viola(reject_levels=0.5, level_weights=2, scale=1.05)


if __name__ == '__main__':
    detect_metal()






