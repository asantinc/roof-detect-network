import os
import subprocess
import pdb
import cProfile

import numpy as np
import cv2
from scipy import misc, ndimage #load images

import get_data
import experiment_settings as settings

MIN_OVERLAP_PERCENT = 0.80


class DataAugmentation(object):
    def transform(self, Xb):
        self.Xb = Xb
        self.random_flip()
        self.random_rotation(10.)
        self.random_crop()
        return self.Xb

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
        # Flip images
        self.Xb_flip_cols = self.Xb[:, :, :, ::-1]
        self.Xb_flip_rows = self.Xb[:, :, ::-1, :]


class ViolaDataSetup(object):
    @staticmethod
    def setup_negative_samples():
        output = open('../viola_jones/bg.txt', 'w')
        for file in os.listdir(settings.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(settings.UNINHABITED_PATH+file+'\n')


    @staticmethod
    def setup_positive_samples(padding=0):
        pad = 'pad'+str(padding)
        metal_n = '../viola_jones/metal_'+pad+'.dat'
        thatch_n = '../viola_jones/thatch_'+pad+'.dat'

        with open(metal_n, 'w') as metal_f, open(thatch_n, 'w') as thatch_f:
            img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.INHABITED_PATH)
            roof_loader = get_data.DataLoader()

            for img_name in img_names_list:
                print 'Processing image: {0}'.format(img_name)
                xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
                img_path = settings.INHABITED_PATH+img_name
        
                roofs, _, _ = roof_loader.get_roofs(xml_path)
                metal_log = list()
                thatch_log = list()

                for roof in roofs:
                    #append roof characteristics separated by single space
                    if roof.roof_type == 'metal':
                        metal_log.append(ViolaDataSetup.get_roof_dat(roof, padding, img_path))
                    elif roof.roof_type == 'thatch':
                        thatch_log.append(ViolaDataSetup.get_roof_dat(roof, padding, img_path))

                if len(metal_log)>0:
                    metal_f.write(ViolaDataSetup.get_dat_string(metal_log, img_path))
                if len(thatch_log)>0:
                    thatch_f.write(ViolaDataSetup.get_dat_string(thatch_log, img_path))


    @staticmethod
    def get_dat_string(roof_list, img_path):
        return img_path+'\t'+str(len(roof_list))+'\t'+'\t'.join(roof_list)+'\n'


    @staticmethod
    def get_roof_dat(roof, padding, img_path):
        '''Return string with roof details including padding if padding can be added
        '''
        try:
            img = misc.imread(img_path)
        except IOError:
            print 'Cannot open '+img_path
        else:
            img_height, img_width, _ = img.shape

        roof_xmin = roof.xmin
        roof_ymin = roof.ymin
        roof_width = roof.width
        roof_height = roof.height

        if roof.xmin-padding > 0:
            roof_xmin -= padding
        if roof.ymin-padding > 0:
            roof_ymin -= padding

        if roof.xmin+roof.width+padding < img_width:
            roof_width += padding
        if roof.ymin+roof.height+padding < img_height:
            roof_height += padding

        return '{0} {1} {2} {3}'.format(roof_xmin, roof_ymin, roof_width, roof_height)

class ViolaDetector(object):

    def __init__(self, num_pos, num_neg, 
            roof_type, 
            detector_paths=None, 
            report_path=None,
            output_folder=None,
            save_imgs=False
            ):

        self.roofs_detected = dict()
        self.roof_detectors = list()
        self.overlap_dict = dict()
        self.save_imgs = save_imgs

        self.detector_paths = detector_paths
        for path in self.detector_paths:
            self.roof_detectors.append(cv2.CascadeClassifier(path))

        #output names
        self.reset_params(num_pos=num_pos, num_neg=num_neg, roof_type=roof_type)

        self.output_folder = output_folder
        self.file_name_default = '''numPos{0}_numNeg{1}_scale{2}_{3}'''.format(num_pos, num_neg, self.scale, roof_type)
        self.report_path = self.output_folder

        #open report file
        self.report_file = self.report_path+'report.txt'
        open(self.report_file, 'w').close()


    def reset_params(self, num_pos=None, num_neg=None, roof_type=None, scale=1.05):
        self.num_pos = num_pos if num_pos is not None else self.num_pos
        self.num_neg = num_neg if num_neg is not None else self.num_neg
        self.roof_type = roof_type if roof_type is not None else self.roof_type
        self.scale = scale
        self.total_roofs = 0
        self.true_positives = 0
        self.all_detections_made = 0  


    def detect_roofs(self, img_name=None, reject_levels=1.3, level_weights=5, scale=None, img_path=None):
        #get image
        img_path = settings.INHABITED_PATH+img_name if img_path is None else img_path+img_name
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if scale is not None:
            self.scale = scale

        #get detections in the image
        roof_detections = list()
        for detector in self.roof_detectors:
            detected_roofs = detector.detectMultiScale(
                        gray,
                        scaleFactor=self.scale,
                        minNeighbors=5)
            roof_detections.append(detected_roofs)
        self.roofs_detected[img_name] = roof_detections


    def report_roofs_detected(self, i, img_name):
        detected_numbers = list()
        for detection_set in self.roofs_detected[img_name]:
            detected_numbers.append(len(detection_set))
        print '*************************** IMAGE:'+str(i)+','+str(img_name)+'***************************'
        print 'Roofs detected: '+str(detected_numbers)+' Total: '+str(sum(detected_numbers))


    def test_viola(self, reject_levels=1.3, level_weights=5, scale=1.05):
        self.reset_params(scale=scale)

        loader = get_data.DataLoader()
        img_names = get_data.DataLoader.get_img_names_from_path(path=settings.INHABITED_PATH)

        for i, img_name in enumerate(list(img_names)):
            self.detect_roofs(img_name=img_name)
            self.report_roofs_detected(i, img_name)

            img_path = settings.INHABITED_PATH+img_name

            try:
                image = misc.imread(img_path)
            except IOError:
                print 'Cannot open '+img_path
            rows, cols, _ = image.shape

            xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
            roof_list, _, _ = loader.get_roofs(xml_path)
            
            self.match_roofs_to_detection(img_name, roof_list, rows, cols)
            self.print_report(img_name)

            if self.save_imgs:
                self.save_detection_img(img_path, img_name, roof_list)

        self.print_report(final_stats=True)


    def match_roofs_to_detection(self, img_name, roof_list, rows=1200, cols=2000):
    #Compare detections to ground truth
        self.overlap_dict[img_name] = list()
        for roof in roof_list:
            if roof.roof_type == self.roof_type:
                self.overlap_dict[img_name].append(roof.max_overlap_single_patch(rows=rows, cols=cols, 
                            detections=self.roofs_detected[img_name]))


    def save_detection_img(self, img_path, img_name, roof_list):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.mark_detections_on_img(img, img_name)
        img = self.mark_roofs_on_img(img, img_name, roof_list)
        self.save_image(img, img_name)


    def print_report(self, img_name='', detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as output:
            if not final_stats:
                curr_detections_made = len(self.roofs_detected[img_name])
                self.all_detections_made += curr_detections_made 

                current_img_roofs = len(self.overlap_dict[img_name])
                self.total_roofs += current_img_roofs
                current_img_true_positives = 0

                for v in self.overlap_dict[img_name]:
                    if v > 0.20:
                        current_img_true_positives += 1
                self.true_positives += current_img_true_positives
                log = 'Image '+img_name+': '+str(current_img_true_positives)+'/'+str(current_img_roofs)+'\n'
                print log
                output.write(log)

            elif final_stats:
                #Print number of false positives, False negatives
                log = ('******************************* RESULTS ********************************* \n'
                    +'Precision: \t'+str(self.true_positives)+'/'+str(self.all_detections_made)+
                    '\n'+'Recall: \t'+str(self.true_positives)+'/'+str(self.total_roofs)+'\n')
                print log
                output.write(log)

                self.true_positive = 0
                self.total_roofs = 0
                self.all_detections_made = 0


    def mark_detections_on_img(self, img, img_name):
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


    def mark_roofs_on_img(self, img, img_name, roof_list):
        for roof in roof_list:
            if roof.roof_type == self.roof_type:
                cv2.rectangle(img,(roof.xmin,roof.ymin),(roof.xmin+roof.width,roof.ymin+roof.height),(0,255,0),2)
        return img


    def save_image(self, img, img_name, output_folder=None):
        if output_folder is not None:
            self.output_folder = output_folder 
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

 
def detect_metal():    
    #set up the data
    #Viola.setup_negative_samples()
    #Viola.setup_positive_samples()
    #train the cascade

    #test the cascade
    num_pos = 209
    num_neg = 700
    roof_type = 'metal'
    
    detectors = ['../viola_jones/cascade_w12_h25_pos393_neg700.xml', 
                '../viola_jones/cascade_w25_h12_pos393_neg700.xml', 
                '../viola_jones/cascade_simple_1500.xml']
    output_folder = '../viola_jones/output_test_time/'

    viola = Viola(num_pos, num_neg, roof_type, detector_paths=detectors, output_folder=output_folder, save_imgs=False)
    viola.test_viola(reject_levels=0.5, level_weights=2, scale=1.05)


def test_single():
    num_pos = 209
    num_neg = 1500

    roof_type = 'metal'
    detectors_2 = ['../viola_jones/cascade_w12_h25_pos393_neg1500.xml', 
                '../viola_jones/cascade_w25_h12_pos393_neg1500.xml', 
                '../viola_jones/cascade_simple_1500.xml']
    output_folder = '../viola_jones/'

    img_name = '0001.jpg'
    img_path = settings.INHABITED_PATH
    xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'

    roof_loader = get_data.DataLoader()
    roof_list, _, _ = roof_loader.get_roofs(xml_path)

    viola = ViolaDetector(num_pos, num_neg, roof_type, detector_paths=detectors_2, output_folder=output_folder)
    img = cv2.imread(img_name, flags=cv2.IMREAD_COLOR)
        
    viola.detect_roofs(img_name=img_name, img_path=img_path, reject_levels=0.5, level_weights=2, scale=1.05)
    viola.match_roofs_to_detection(img_name, roof_list)
    viola.print_report(img_name)


if __name__ == '__main__':
    #cProfile.run(detect_metal())
    #test_single()
    ViolaDataSetup.setup_positive_samples(padding=15)





