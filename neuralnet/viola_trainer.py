import os
import subprocess
import pdb
import math
import random
import getopt
import sys
import csv

import numpy as np
import cv2
import cv
from scipy import misc, ndimage #load images

import get_data
from timer import Timer

class ViolaTrainer(object):
    @staticmethod
    def train_cascade(vec_files=None, feature_type='haar', max_false_alarm_rate=0.5, stages=20, minHitRate=0.99999, roof_type=None, padding=-1):
        cascades = list()
        roof_type = roof_type
        for vec_file in vec_files:
            vec_type = vec_file[:5] if roof_type == 'metal' else vec_file[:6]
            if (roof_type is None or vec_type  == roof_type):
                #get cascade parameters from file name
                w = vec_file[-10:-8]
                h = vec_file[-6:-4]
                index = vec_file.find('num') 
                assert index != -1
                sample_num = vec_file[index+3:-12]
                assert int(float(sample_num)) > 0

                print 'Training with vec file: {0}'.format(vec_file)
                cascade_folder = '../viola_jones/cascade_{0}_FA{1}_{2}/'.format(vec_file[:-4], max_false_alarm_rate, feature_type)
                cascades.append(cascade_folder+'cascade.xml')
                mkdir_cmd = 'mkdir {0}'.format(cascade_folder)
                try:
                    subprocess.check_call(mkdir_cmd, shell=True)
                except Exception as e:
                    print e
                
                cmd = list()
                cmd.append('/usr/bin/opencv_traincascade')
                cmd.append('-data {0}'.format(cascade_folder))
                cmd.append('-vec ../viola_jones/vec_files/{0}'.format(vec_file))
                cmd.append('-bg ../viola_jones/bg.txt')
                cmd.append('-numStages {0}'.format(stages)) 
                cmd.append('-minHitRate {0}'.format(minHitRate))
                if feature_type != 'haar':
                    cmd.append('-featureType LBP')
                cmd.append('-maxFalseAlarmRate {0}'.format(max_false_alarm_rate))
                cmd.append('-precalcValBufSize 1024 -precalcIdxBufSize 1024')
                numPos = int(float(sample_num)*.8)
                cmd.append('-numPos {0} -numNeg {1}'.format(numPos, numPos*2))
                cmd.append('-w {0} -h {1}'.format(w, h))
                train_cmd = ' '.join(cmd)
                try:
                    print train_cmd
                    subprocess.check_call(train_cmd, shell=True)
                except Exception as e:
                    print e
        return cascade_folder 


def main(max_false_alarm=0.2, feature_type=None):
    #TRAINING CASCADE
    assert feature_type is not None
    no_details = True
    roof_type = ''  
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:t:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-f':
            v = arg if arg.endswith('.vec') else arg+'.vec'
            no_details = False
        elif opt == '-t':
            roof_type = arg
    roof_type = 'metal' if v[:1]=='m' else 'thatch'

    if no_details:
        v = raw_input('Enter vec file: ')
        t = raw_input('Type of roof: ' )
        roof_type = 'metal' if t=='m' else 'thatch'
    vecs = [v]
    with Timer() as t:
        folder_path = ViolaTrainer.train_cascade(vec_files=vecs, max_false_alarm_rate=max_false_alarm, feature_type=feature_type, roof_type=roof_type)
    open('{0}training_secs_{1}'.format(folder_path, t.secs)).close()


if __name__ == '__main__':
    main(max_false_alarm=0.3, feature_type='LBP')
