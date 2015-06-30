import sys
import getopt
import os
import subprocess

from scipy import misc

import convolution
import experiment_settings as settings
from get_data import DataLoader, Roof
from viola_detector import ViolaDetector


#TODO: depending on resolution this will be affected... 
STEP_SIZE = settings.PATCH_H/4

class Pipeline(object):
	def __init__(self, test_files=None):
		#image_files
                print test_files
		if test_files is None:
			self.test_fnames = DataLoader.get_img_names_from_path(path = settings.TEST_PATH)
			print 'hi'
			print self.test_fnames
		else:
			self.test_fnames = test_files
                assert self.test_fnames is not None
	        print self.test_fnames

	   	#OUTPUT FOLDER: create it if it doesn't exist
		out_name = raw_input('Name of output folder: ')
		assert out_name != ''
		self.out_path = '../output/pipeline/{0}'.format(out_name)
		if not os.path.isdir(out_name):
			subprocess.check_call('mkdir {0}'.format(self.out_path), shell=True)
		print 'Will output files to: {0}'.format(self.out_path)

		#DETECTORS
		self.detector_paths = list()
		while True:
	                cascade_name = raw_input('Cascade file to use: ' )
        	        detector = settings.CASCADE_PATH+cascade_name+'.xml'
            	        if cascade_name == '':
				break
			self.detector_paths.append(detector)
		print 'Using detectors: '+'\t'.join(self.detector_paths)

		#create report file
		self.report_path = self.out_path+'report.txt'
		with open(self.report_path, 'w') as report:
			report.write('\t'.join(self.detector_paths))

		self.viola = ViolaDetector(detector_paths=self.detector_paths, output_folder=self.out_path, save_imgs=True)
		#the output should be a picture with the windows marked!
		self.neural_net = self.setup_neural_net()


	def run(self):
		'''
		1. Find contours using Viola and Jones for candidate roofs
		2. Within each contour, use neural network to classify sliding patches
		3. Net returns a list of the roof coordinates of each type - saved in roof_coords
		'''
		self.thatch_masks = dict()
		self.metal_masks = dict()
		self.roof_coords = dict()
		self.all_contours = dict()
		
		for img_name in self.test_fnames:
			print 'Pre-processing image: {0}'.format(img_name)
			try:
				image = misc.imread(settings.TEST_PATH+img_name)
			except IOError:
				print 'Cannot open '+img_path
			rows, cols, _ = image.shape
			
			thatch_mask = np.zeros((rows, cols), dtype=boolean)

			self.process_viola(img_name, image, verbose=True)

			metal_coords, thatch_coords = self.sliding_convolution(img, contours)
			roofs_detected[img_name] = (metal_coords, thatch_coords)


	def setup_neural_net(self):
		return ValueError('TO BE IMPLEMENTED')


	def process_viola(self, img_name, image, verbose=False):
		'''
		Find candidate roof contours using Viola for all types of roof
		'''
		#returns list with as many lists of detections as the detectors we have passed
		detections = self.viola.detect_roofs(img_name=img_name, img_path=settings.TEST_PATH)
		if verbose:
			self.viola.mark_detections_on_img(self, img=image, img_name=img_name)

		#get the mask and the contours for the detections
		detection_mask = self.viola.get_patch_mask(img_name=img_name, rows=rows, cols=cols)
		self.all_contours[img_name] = self.viola.get_detection_contours(self, detection_mask, img_name)


	def sliding_convolution(self, img, contours, step_size):
		#for each contour, do a sliding window detection
		for cont in contours:
			x,y,w,h = cv2.boundingRect(cont)
	        vert_patches = ((h - settings.PATCH_H) // self.step_size) + 1  #vert_patches = h/settings.PATCH_H

	        #get patches along roof height
	        for vertical in range(vert_patches):
	            y_pos = roof.ymin+(vertical*self.step_size)
	            #along roof's width
	            self.get_horizontal_patches((x,y,w,h), img=img, y_pos=y_pos)

	        #get patches from the last row also
	        if (h % settings.PATCH_W>0) and (h > settings.PATCH_H):
	            leftover = h-(vert_patches*settings.PATCH_H)
	            y_pos = y_pos-leftover
	            self.get_horizontal_patches((x,y,w,h), img=img, y_pos=y_pos)

		return metal_coords, thatch_coords


	def get_horizontal_patches(self, patch=None, img=None, roof=None, y_pos=-1):
	    '''Get patches along the width of a patch for a given y_pos (i.e. a given height in the image)
	    '''
	    roof_type = settings.METAL if roof.roof_type=='metal' else settings.THATCH

	    h, w = roof.get_roof_size()
	    hor_patches = ((w - settings.PATCH_W) // self.step_size) + 1  #hor_patches = w/settings.PATCH_W

	    for horizontal in range(hor_patches):
	        x_pos = roof.xmin+(horizontal*self.step_size)
	        candidate = img[x_pos:x_pos+settings.PATCH_W, y_pos:y_pos+settings.PATCH_H]
	        #do network detection


if __name__ == '__main__':
	#to detect a single image pass it in as a parameter
	#else, pipeline will use the files in settings.TEST_PATH folder
	try:
		opts, args = getopt.getopt(sys.argv[1:], "f:")
	except getopt.GetoptError:
		print 'Command line error'
		sys.exit(2) 
	test_file = None
	for opt, arg in opts:
		if opt == '-f':
            test_file = arg
            test_files = list()
            test_files.append(test_file)
	pipe = Pipeline(test_files=test_files)
	#dictionaries accessed by file_name
	metal_detections, thatch_detections = pipe.run()



