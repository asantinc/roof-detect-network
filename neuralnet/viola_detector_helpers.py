from viola_detector import ViolaDetector 
from reporting import Evaluation, Detections
import getopt
import sys
from collections import defaultdict
import csv
import pdb

import utils


def pickle_neural_true_false_positives():
    #Getting patches for neural network
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-c':
            combo_f = arg
    assert combo_f is not None
    detectors = get_detectors(combo_f)
    viola = NeuralViolaDetector(detector_names=detectors, save_imgs=False, old_detector = False, neural=True)
    if len(detectors['metal']) > 0:
        roof_type = 'metal'
    elif len(detectors['thatch']) > 0:
        roof_type = 'thatch'
    else:
        raise ValueError('No detector found in combo{0}'.format(combo_f))
    viola.get_neural_training_data(save_false_pos = True, roof_type=roof_type, detector_name='combo'+combo_f) 


def get_detectors(combo_f):
    detectors = dict()
    detector_file = utils.get_path(viola=True, params=True)+'combo'+str(combo_f)+'.csv'
    detectors = defaultdict(list)
    with open(detector_file, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        for line in r:
            if len(line) < 2:
                continue
            if line[0] == 'metal':
                detectors['metal'].append(line[1].strip())
            elif line[0] == 'thatch':
                detectors['thatch'].append(line[1].strip())
            else:
                raise ValueError("Unknown detector type {0}".format(line[0]))
    return detectors


def get_all_combos():
    path = utils.get_path(viola=True, params=True) 
    detector_list = list()
    combo_f_names = list()

    for f_name in os.listdir(path):
        if os.path.isfile(path+f_name) and f_name.startswith('combo') and ('equalized') in f_name:
            combo_f = f_name[5:7]
            if combo_f.endswith('.'):
                combo_f = combo_f[:1]
            
            detector_list.append(get_detectors(combo_f))
            combo_f_names.append(combo_f)
    return detector_list, combo_f_names 


def testing_detectors(group=None, original_dataset=False, all=False, data_fold=utils.VALIDATION):
    '''Test either a single detector or all detectors in the combo files
    '''
    #TESTING ONLY ONE DETECTOR that must be passed in as an argument  
    if all == False:
        combo_f = None
        try:
            opts, args = getopt.getopt(sys.argv[1:], "c:")
        except getopt.GetoptError:
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-c':
                combo_f = arg
        assert combo_f is not None
        detectors = get_detectors(combo_f)
        detector_list = [detectors]
        combo_f_names = [combo_f]
    else:#get all detector combos
        detector_list, combo_f_names =  get_all_combos()

    #need to detect on validation set. To make it easier to digest, initially we also looked at training set
    for detector, combo_f_name in zip(detector_list, combo_f_names):
        in_path = utils.get_path(viola=True, input_or_output='input', data_fold=data_fold)
        out_path = utils.get_path(viola=True, input_or_output='output', data_fold=data_fold)

        if group=='group_rectangles':    
            folder_name = 'group_rect_combo'+combo_f_name+'/'
        elif group=='group_bounding':
            folder_name = 'group_bound_combo'+combo_f_name+'/'
        else:
            folder_name = 'combo'+combo_f_name+'/'

        viola = ViolaDetector(out_folder_name=folder_name, out_path=out_path, in_path=in_path, 
                                                detector_names=detector, save_imgs=False, old_detector = False)
        viola.detect_roofs_in_img_folder(group=group, reject_levels=0.5, level_weights=2, scale=1.05)


def check_cascade_status():
    '''Check the status of all of the cascades
    '''
    casc_path = '../viola_jones/'
    for f_name in os.listdir(casc_path):
        if os.path.isdir(casc_path+f_name) and f_name.startswith('cascade_'):
            if os.path.isfile(casc_path+f_name+'/cascade.xml'):
                print '{0}\t\t\t done'.format(f_name)
            else:
                print '{0}\t\t\t MISSING'.format(f_name)


def set_up_basic_combos():
    '''Set up combo with a single detector per combo
    '''
    raise ValueError('This needs fixing')
    casc_path = '../viola_jones/'
    out_path = '../viola_jones/detector_combos/combo'
    combo_num = 0
    f_names = dict() #roof_type, equalized, augmented
    f_names['metal'] = dict()
    f_names['thatch'] = dict()
    f_names['metal']['equalized']=defaultdict(set)
    f_names['metal']['not_equalized'] =defaultdict(set)
    f_names['thatch']['equalized']=defaultdict(set)
    f_names['thatch']['not_equalized']=defaultdict(set)

    for f_name in os.listdir(casc_path):
        if os.path.isdir(casc_path+f_name) and f_name.startswith('cascade_') and ('equalized' in f_name):
            if os.path.isfile(casc_path+f_name+'/cascade.xml'):
                try:
                    roof_type = 'metal' if 'metal' in f_name else 'thatch'
                    equalized = 'not_equalized' if 'not_equalized' not in f_name else 'equalized'
                    augmented = 'augm1' if 'augm1' in f_name else 'augm0'
                    f_names[roof_type][equalized][augmented].add(f_name)
                except KeyError:
                    pdb.set_trace()
                detector_name = f_name[8:]
                with open('{0}{1}.csv'.format(out_path, combo_num), 'w') as f:
                    if detector_name.startswith('metal'):
                        d_type = 'metal'
                    elif detector_name.startswith('thatch'):
                        d_type = 'thatch'
                    else:
                        raise ValueError('Unknown roof type for cascade')

                    f.write('{0}, {1}'.format(d_type, detector_name))
                combo_num += 1
            else:
                print 'Could not process incomplete: {0}'.format(f_name)
    #set up the rectangular and square detectors together 
    for roof_type in ['metal', 'thatch']:
        for equalized in ['equalized', 'not_equalized']:
            for augm in ['augm1', 'augm0']:
                detectors = f_names[roof_type][equalized][augm]
                if len(detectors) > 1:
                    #write a new combo file
                    with open('{0}{1}.csv'.format(out_path, combo_num), 'w') as f:
                        log_to_file = ''
                        for d in detectors:
                            detector_name = d[8:]
                            if detector_name.startswith('metal'):
                                d_type = 'metal'
                            elif detector_name.startswith('thatch'):
                                d_type = 'thatch'
                            else:
                                raise ValueError('Unknown roof type for cascade')
                            log_to_file += '{0}, {1}\n'.format(d_type, detector_name)
                        f.write(log_to_file)
                        combo_num += 1
                else:
                    print 'Only one detector found: {0}'.format(detectors)



if __name__ == '__main__':
    testing_detectors(all=False, original_dataset=True, data_fold=utils.TESTING)
    
