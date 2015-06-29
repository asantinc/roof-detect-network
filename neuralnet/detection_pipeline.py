import train_convolution
import experiment_settings as settings
from get_data import DataLoader
from viola_detector import ViolaDetector

# get the right viola cascade
# detect roofs: metal and thatch --> return a set of windows
# do we want to find the contour?
# Decide what to do with the windows
	#maybe we can do a sliding cascade thing?
	#should we do it at different scales?
#if a patch is a 'yes'


if __name__ == '__main___':
	#image_files
	test_fnames = DataLoader.get_img_names_from_path(path = settings.TEST_PATH)
	
	#cascades
	cascade_name = raw_input('Cascade file to use: ' )
	cascade_path =  settings.CASCADE_PATH+cascade_name+'.xml'

	#list of detectors
	#TODO: determine what these should be
	#Remember you need metal and thatch detectors
	current_cascade = 'cascade_1'
	detectors_metal = ['../viola_jones/{0}/cascade.xml'.join(current_cascade)]
    out_metal = '../output/{0}/.join(current_cascade)'
    viola_metal = ViolaDetector(detector_paths=detectors_metal, output_folder=out_metal, save_imgs=True)

	current_cascade = 'cascade_1'
	detectors_metal = ['../viola_jones/{0}/cascade.xml'.join(current_cascade)]
    out_metal = '../output/{0}/.join(current_cascade)'
    viola_metal = ViolaDetector(detector_paths=detectors_metal, output_folder=out_metal, save_imgs=True)

	for img_name in test_fnames:
		#for each each detector
		#do detection on current image
		#returns list with as many lists of detections as the detectors we have passed
		detections_metal = viola_metal.detect_roofs(img_name=img_name, img_path=test_fnames)
		#Need to check how the detections from the different detectors are returned

		#TODO: for each window detected, check what the neural network says it is
		#first don't merge it because we want to know if viola and jones does a terrible job
		
		













