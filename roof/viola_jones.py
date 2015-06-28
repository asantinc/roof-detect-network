import os
import subprocess
import pdb
import cProfile
import math

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import get_data
from get_data import DataAugmentation
import experiment_settings as settings


class ViolaDataSetup(object):
    @staticmethod
    def vec_file_samples():
        '''Save a vec file for every .dat file found in ../viola_jones/all_dat/unprocessed/
        '''
        vec_info = list()
        dat_path = '../viola_jones/all_dat/unprocessed/'
        vec_path = '../viola_jones/vec_files/'
        bg_file = '../viola_jones/bg.txt'
        for file_name in os.listdir(dat_path):
            if file_name.endswith('.dat'):
                vec_file_name = file_name[:-3]+'vec'
                vec_file = vec_path+vec_file_name
                sample_num = raw_input('Sample nums for {0}: '.format(file_name))
                w = raw_input('Width of {0}: '.format(file_name))
                h = raw_input('Height of {0}: '.format(file_name))

                vec_info.append( (vec_file_name, int(float(sample_num)), int(float(w)), int(float(h))) )

                dat_file = dat_path+file_name
                sample_cmd ='opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_file, bg_file, vec_file, sample_num, w, h)
                try:
                    subprocess.check_call(sample_cmd, shell=True)
                    move_cmd = 'mv {0} ../viola_jones/all_dat/'.format(dat_file)
                    subprocess.check_call(move_cmd, shell=True)
                except Exception as e:
                    print e
        return vec_info


    @staticmethod
    def vec_file_single(dat_file=''):
        '''Produce vec file for given dat file
        '''
        bg_path = '../viola_jones/bg.txt'
        vec_file = dat_file[:-4]+'.vec'
        vec_path = '../viola_jones/vec_files/{0}'.format(vec_file)
        dat_path = '../viola_jones/all_dat/unprocessed/{0}'.format(dat_file)
        sample_num = raw_input('Sample nums for {0}: '.format(dat_file[:-4]))
        w = raw_input('Width of {0}: '.format(dat_file[:-4]))
        h = raw_input('Height of {0}: '.format(dat_file[:-4]))

        vec_cmd ='opencv_createsamples -info {0} -bg {1} -vec {2} -num {3} -w {4} -h {5}'.format(dat_path, bg_path, vec_path, sample_num, w, h)
        try:
            subprocess.check_call(vec_cmd, shell=True)
        except Exception as e:
            print e 
        return [(vec_file, sample_num, w, h)]     


    @staticmethod
    def setup_negative_samples():
        '''Write file with info about location of negative examples
        '''
        output = open('../viola_jones/bg.txt', 'w')
        for file in os.listdir(settings.UNINHABITED_PATH):
            if file.endswith('.jpg'):
                output.write(settings.UNINHABITED_PATH+file+'\n')
    

    @staticmethod
    def transform_roofs(padding=5):
        ''' Save augmented jpgs of data with padding, rotations and flips
        '''
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=settings.INHABITED_PATH)
        roof_loader = get_data.DataLoader()

        for img_id, img_name in enumerate(img_names_list):
            print 'Processing image: {0}'.format(img_name)
            xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
            img_path = settings.INHABITED_PATH+img_name
    
            roofs, _, _ = roof_loader.get_roofs(xml_path)
            metal_log = list()
            thatch_log = list()

            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except IOError:
                print 'Cannot open '+img_path
            else:
                for roof_id, roof in enumerate(roofs):
                    print 'Processing image {0}: roof {1}'.format(img_id, roof_id)

                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, padding, img_path)

                    roof_img = np.copy(gray[roof.ymin:roof.ymin+roof.height,roof.xmin:roof.xmin+roof.width])

                    if roof.roof_type == 'metal':
                        general_path = '../viola_jones/data/metal/img{0}_'.format(img_id)
                    else:
                        general_path = '../viola_jones/data/thatch/img{0}_'.format(img_id)

                    for rotation in range(4):
                        if rotation == 0:
                            pass
                        elif rotation > 0:
                            roof_img = DataAugmentation.rotateImage(roof_img)
                        patch_path = '{0}id{1}_rot{2}'.format(general_path, roof_id, rotation)
                        #save the patch as is
                        cv2.imwrite('{0}.jpg'.format(patch_path), roof_img)
                        DataAugmentation.flip_save(roof_img, patch_path)


    @staticmethod
    def setup_positive_samples_full_image(roof_type=None, padding=5):
        '''Return .dat files containing positions and sizes of roofs in training images
        '''
        metal = [list() for i in range(3)]
        metal[0] = '../viola_jones/all_dat/unprocessed/metal_{0}_tall_augment.dat'.format(padding)
        metal[1] = '../viola_jones/all_dat/unprocessed/metal_{0}_square_augment.dat'.format(padding)
        metal[2] = '../viola_jones/all_dat/unprocessed/metal_{0}_wide_augment.dat'.format(padding)

        thatch_n = '../viola_jones/all_dat/unprocessed/thatch_{0}_augment.dat'.format(padding)

        with open(metal[0], 'w') as metal_f_tall, open(metal[1], 'w') as metal_f_square, open(metal[2], 'w') as metal_f_wide, open(thatch_n, 'w') as thatch_f:
            path = ''
            if roof_type == 'metal':
                path = '../viola_jones/data/metal/'
                img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
            elif roof_type == 'thatch':
                path = '../viola_jones/data/thatch/'
                img_names_list = get_data.DataLoader().get_img_names_from_path(path='../viola_jones/data/thatch/')

            for img_name in img_names_list:
                img_path = path+img_name
                try:        
                    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                except IOError:
                    print 'Cannot open '+img_path
                else:
                    height, width, _ = img.shape

                    #append roof characteristics separated by single space
                    position_string = '{0} {1} {2} {3}'.format(0, 0, width, height)
                    log_to_file = '{0}\t{1}\t{2}\n'.format('../../'+img_path, 1, position_string)

                    #add to .dat file depending on roof time and on ratio (if metal)
                    ratio = float(width)/height
                    if roof_type == 'metal':
                        if ratio > 1.15:
                            metal_f_wide.write(log_to_file)
                        elif ratio < 0.80:
                            metal_f_tall.write(log_to_file)
                        else:
                            metal_f_square.write(log_to_file)
                    else:
                         thatch_f.write(log_to_file)
            return {'thatch':[thatch_n], 'metal': metal}
        

    @staticmethod
    def setup_positive_samples(padding=0, path=settings.INHABITED_PATH):
        '''
        Return .dat files containing positions and sizes of roofs in training images
        This uses the info about the Inhabited roofs but adds padding also and produces .Dat
        file with samples embedded in the full images
        '''
        pad = 'pad'+str(padding)
        metal = [list() for i in range(3)]
        metal[0] = '../viola_jones/metal_'+pad+'_tall.dat'
        metal[1] = '../viola_jones/metal_'+pad+'_square.dat'
        metal[2] = '../viola_jones/metal_'+pad+'_wide.dat'
        thatch_n = '../viola_jones/thatch_'+pad+'.dat'

        with open(metal[0], 'w') as metal_f_tall, open(metal[1], 'w') as metal_f_square, open(metal[2], 'w') as metal_f_wide, open(thatch_n, 'w') as thatch_f:
            img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
            roof_loader = get_data.DataLoader()

            for img_name in img_names_list:
                print 'Processing image: {0}'.format(img_name)
                xml_path = settings.INHABITED_PATH+img_name[:-3]+'xml'
                img_path = settings.INHABITED_PATH+img_name
        
                roofs, _, _ = roof_loader.get_roofs(xml_path)
                metal_log = [list() for i in range(3)]
                thatch_log = list()

                for roof in roofs:
                    #append roof characteristics separated by single space
                    if padding > 0:
                        roof = DataAugmentation.add_padding(roof, img_path)

                    if roof.roof_type == 'metal':
                        ViolaDataSetup.get_roof_dat(metal_log, roof)
                    elif roof.roof_type == 'thatch':
                        ViolaDataSetup.get_roof_dat(thatch_log, roof)

                if len(metal_log[0]) > 0:
                    metal_f_tall.write(ViolaDataSetup.get_dat_string(metal_log[0], img_path))
                if len(metal_log[1]) > 0:
                    metal_f_square.write(ViolaDataSetup.get_dat_string(metal_log[1], img_path))
                if len(metal_log[2]) > 0:
                    metal_f_wide.write(ViolaDataSetup.get_dat_string(metal_log[2], img_path))
                if len(thatch_log) > 0:
                    thatch_f.write(ViolaDataSetup.get_dat_string(thatch_log, img_path))


    @staticmethod
    def get_dat_string(roof_list, img_path):
        '''Return string formatted for .dat file
        '''
        return img_path+'\t'+str(len(roof_list))+'\t'+'\t'.join(roof_list)+'\n'


    @staticmethod
    def get_roof_dat(roof_lists, roof):
        '''Return string with roof positive and size. Roof is added to list depending on width/height ratio
        '''
        string_to_add = '{0} {1} {2} {3}'.format(roof.xmin, roof.ymin, roof.width, roof.height)
        #if we care about roof size, add the roofs to the right list depending on their height/width ratio
        ratio = 0
        if roof.roof_type == 'metal':
            aspect_ratio = float(roof.width)/(roof.height)
            if (aspect_ratio > 1.5):                       #TALL ROOF
                roof_lists[2].append(string_to_add)
                pdb.set_trace()
            elif aspect_ratio >= 0.75 and ratio <= 1.5:    #SQUARE
                roof_lists[1].append(string_to_add)
            elif aspect_ratio < 0.75:                      #WIDE ROOF
                roof_lists[0].append(string_to_add)
        else:
            roof_lists.append(string_to_add)


    @staticmethod
    def produce_imgs_simple_detector():
        ''' Squares and rectangles to use as simple detectors instead of the roof images
        '''
        path = '../viola_jones/data/data_simple/'
        angle = 45  

        background = np.zeros((200,120))
        background[10:190, 10:110] = 1
        blur_back = cv2.GaussianBlur(background, (21,21), 0)
        misc.imsave('simple_{0}vert_rect.jpg'.format(path), blur_back)

        background = np.zeros((120,200))
        background[10:110, 10:190] = 1
        blur_back = cv2.GaussianBlur(background, (21,21), 0)
        misc.imsave('simple_{0}horiz_rect.jpg'.format(path), blur_back)

        rotated_blur = ndimage.rotate(blur_back, 45)
        misc.imsave('simple_{0}diag_rect.jpg'.format(path), rotated_blur)

        #square
        square = np.zeros((200,200))
        square[10:190, 10:190] = 1
        blur_square = cv2.GaussianBlur(square, (21,21), 0)
        misc.imsave('simple_{0}square.jpg'.format(path), blur_square)

        rot_square = ndimage.rotate(blur_square, 45)
        misc.imsave('simple_{0}diag_square.jpg'.format(path), rot_square)


    @staticmethod
    def setup_positive_samples_simple_detector(roof_type='metal', padding=5):
        ''' Create dat files from jpgs, where the samples are just a repetition of a single jpgs
        '''
        path = '../viola_jones/data/data_simple/'
        out_path = '../viola_jones/all_dat/unprocessed/'
        img_names_list = get_data.DataLoader().get_img_names_from_path(path=path)
        
        #produce repeated samples of each .jpg
        for img_name in img_names_list:
            print 'Processing {0}'.format(img_name)
            img_path = path+img_name
            try:        
                img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                out_file = open(out_path+img_name[:-3]+'dat', 'w')
            except IOError:
                print 'Cannot do IO'
            else:
                height, width, _ = img.shape
                for i in range(100):
                    #append roof characteristics separated by single space
                    position_string = '{0} {1} {2} {3}'.format(0, 0, width, height)
                    log_to_file = '{0}\t{1}\t{2}\n'.format('../../'+img_path, 1, position_string)
                    out_file.write(log_to_file)
                 

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
        
        #report detectors being used
        try:
            report = open(self.report_file, 'w')
            report.write('****************** DETECTORS USED ******************\n')
            report.write('\n'.join(detector_paths))
            report.write('\n')

        except IOError as e:
            print e
        
        else:
            report.close()


    @staticmethod
    def train_cascade(vec_info=None, stages=20, minHitRate=0.99999):
        #Train using the vec files and info about them
        cascades = list()
        for (vec_file, sample_num, w, h) in vec_info:
            print 'Training with vec file: {0}'.format(vec_file)
            cascade_folder = '../viola_jones/cascade_{0}/'.format(vec_file[:-4])
            cascades.append(cascade_folder+'cascade.xml')
            mkdir_cmd = 'mkdir {0}'.format(cascade_folder)
            try:
                subprocess.check_call(mkdir_cmd, shell=True)
            except Exception as e:
                print e
            cmd = list()
            cmd.append('opencv_traincascade')
            cmd.append('-data {0}'.format(cascade_folder))
            cmd.append('-vec ../viola_jones/vec_files/{0}'.format(vec_file))
            cmd.append('-bg ../viola_jones/bg.txt')
            cmd.append('-numStages {0}'.format(stages)) 
            cmd.append('-minHitRate {0}'.format(minHitRate))
            numPos = int(float(sample_num)*.8)
            cmd.append('-numPos {0} -numNeg {1}'.format(numPos, numPos*2))
            cmd.append('-w {0} -h {1}'.format(w, h))
            train_cmd = ' '.join(cmd)
            try:
                print train_cmd
                subprocess.check_call(train_cmd, shell=True)
            except Exception as e:
                print e
        return cascades


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
        for i, detector in enumerate(self.roof_detectors):
            print 'Detecting with detector: '+str(i)
            detected_roofs = detector.detectMultiScale(
                        gray,
                        scaleFactor=self.scale,
                        minNeighbors=5)
            group_detected_roofs, weights = cv2.groupRectangles(np.array(detected_roofs).tolist(), 0, 5)
            roof_detections.append(group_detected_roofs)
        self.roofs_detected[img_name] = roof_detections
        print 'Detection is done'


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


    def get_patch_mask(self, img_name, rows=1200, cols=2000):
        patch_mask = np.zeros((rows, cols), dtype=bool)
        patch_area = 0
        for i, cascade in enumerate(self.roofs_detected[img_name]):
            for (x,y,w,h) in cascade:
                patch_mask[y:y+h, x:x+w] = True
                patch_area += w*h
        return patch_mask, patch_area


    def get_detection_contours(self, patch_path, img_name):
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
        cv2.imwrite(bounding_path, output_bounds)

        
    def match_roofs_to_detection(self, img_name, roof_list, rows=1200, cols=2000):
        #Compare detections to ground truth
        self.overlap_dict[img_name] = list()
        patch_mask, patch_area = self.get_patch_mask(img_name)
        patch_location = self.output_folder+img_name+'_mask.jpg'
        #cv2.imwrite(patch_location, patch_mask.astype(np.int8))
        misc.imsave(patch_location, patch_mask)

        detection_contours = self.get_detection_contours(patch_location, img_name)
        for roof in roof_list:
            if roof.roof_type == self.roof_type:
                #self.overlap_dict[img_name].append(roof.max_overlap_single_patch(rows=rows, cols=cols,detections=self.roofs_detected[img_name]))
                self.overlap_dict[img_name].append(roof.check_overlap_total(patch_mask, patch_area, rows, cols))


    def save_detection_img(self, img_path, img_name, roof_list):
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        img = self.mark_detections_on_img(img, img_name)
        img = self.mark_roofs_on_img(img, img_name, roof_list)
        self.save_image(img, img_name)


    def print_report(self, img_name='', detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as output:
            if not final_stats:
                curr_detections_made = sum([len(detections) for detections in self.roofs_detected[img_name]])
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
                pdb.set_trace()
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


 
def detect(detector_paths=None, output='', roof_type=None):    
    #test the cascade
    num_pos = 0
    num_neg = 0
    roof_type = roof_type
    viola = ViolaDetector(num_pos, num_neg, roof_type, detector_paths=detectors, output_folder=output, save_imgs=True)
    viola.test_viola(reject_levels=0.5, level_weights=2, scale=1.05)


if __name__ == '__main__':
    #ViolaDataSetup.setup_positive_samples(padding=5)
    #ViolaDataSetup.transform_roofs(padding=5)
    #ViolaDataSetup.setup_positive_samples_full_image()
    #ViolaDataSetup.setup_positive_samples_simple_detector()

    ## training stuff from scratch
    # ViolaDataSetup.produce_imgs_simple_detector()
    # 
    # cascades = ViolaDataSetup.train_cascade(vec_info)
    # cascades_trained = open('latest_cascades.txt', 'w')
    # cascades_trained.write('/n'.join(cascades))

    #dat_files_dict = ViolaDataSetup.setup_positive_samples_full_image(roof_type='thatch', padding=5)
    #thatch_file_name = dat_files_dict['thatch']
    #vec_info_list = ViolaDataSetup.vec_file_single(dat_file=thatch_file_name[0])
    cascades = ViolaDetector.train_cascade(vec_info=[('thatch_5_augment.vec', 11000, 12, 12)])
    detect(detector_paths=cascades, output='../viola_jones/all_output/thatch/', roof_type='thatch')

    #train and detect thatched roofs



