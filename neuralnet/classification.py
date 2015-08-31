import utils
import os
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from reporting import Detections, Evaluation
from sklearn.metrics import auc, classification_report, accuracy_score
import pdb
import math
import cPickle as pickle

DEBUG = False

class Classification(object):
    def __init__(self, img_names, out_path, correct_roofs, method):
        if DEBUG:
            self.img_names = img_names[:1]
        else:
            self.img_names = img_names

        if DEBUG:
            self.uninhabited = [img_name for img_name in os.listdir(utils.UNINHABITED_PATH)][:1]
        else:
            self.uninhabited = [img_name for img_name in os.listdir(utils.UNINHABITED_PATH)]

        self.correct_roofs = correct_roofs
        self.out_path = out_path
        self.probs = dict()
        self.detections = dict()
        self.method = method

        for roof in utils.ROOF_TYPES:
            self.probs[roof] = dict()
            self.detections[roof] = dict()
        
        #Y_TRUE is always the same, but Y_PRED will vary depending on threshold
        self.y_true = dict()
        for roof_type in utils.ROOF_TYPES:
            self.y_true[roof_type] = np.zeros((len(self.img_names)+len(self.uninhabited))) 
            for i, img_name in enumerate(self.img_names):
                if len(self.correct_roofs[roof_type][img_name]) > 0:
                    self.y_true[roof_type][i] = 1

        self.y_true_any_roof = np.array((self.y_true['thatch']+self.y_true['metal']) > 0, dtype=int)


    def set_detections(self, detections, img_name):
        for roof_type, detects in detections.iteritems():
            self.detections[roof_type][img_name] = np.array(detects)

    def set_probs(self, probs, img_name): 
        #TODO: update the probs per image
        for roof_type, detects in probs.iteritems():
            self.probs[roof_type][img_name] = np.array(probs[roof_type])  


    def get_accuracies(self):
        #find the unique prob points
        self.unique_probs = set()
        self.unique_probs.update([1.])
        for roof_type in utils.ROOF_TYPES:
            for img_name in self.img_names:
                probs = [int(100*p) for p in list(self.probs[roof_type][img_name])]
                probs = [(float(p)/100) for p in probs]
                self.unique_probs.update((probs))

        self.unique_probs = sorted(self.unique_probs)
        
        self.y_pred = dict()
        self.accuracy = dict()

        for thres in self.unique_probs:
            self.y_pred[thres] = dict()
            print 'TRESHOLD: {}'.format(thres)
            for roof_type in utils.ROOF_TYPES:
                if roof_type not in self.accuracy:
                    self.accuracy[roof_type] = list()
                
                self.y_pred[thres][roof_type] = np.zeros((self.y_true[roof_type].shape[0]))
                for i, img_name in enumerate(self.img_names+self.uninhabited):
                    filtered_detections = self.detections[roof_type][img_name][np.array(self.probs[roof_type][img_name]) > thres]
                    if len(filtered_detections)>0:
                        self.y_pred[thres][roof_type][i] = 1
                self.accuracy[roof_type].append(classification_report(self.y_true[roof_type], self.y_pred[thres][roof_type]))  

    def get_accuracies_any_rooftype(self):
        #find the unique prob points
        self.unique_probs = set()
        self.unique_probs.update([1.])
        for roof_type in utils.ROOF_TYPES:
            for img_name in self.img_names:
                probs = [int(100*p) for p in list(self.probs[roof_type][img_name])]
                probs = [(float(p)/100) for p in probs]
                self.unique_probs.update((probs))

        self.unique_probs = sorted(self.unique_probs)
        self.accuracy_anyRoof = [] 
        self.y_pred_any_roof = dict()
        for thres in self.unique_probs:
            self.y_pred[thres] = list() 
            print 'TRESHOLD: {}'.format(thres)
                
            self.y_pred_any_roof[thres] = np.zeros((self.y_true_any_roof.shape[0]))
            for i, img_name in enumerate(self.img_names+self.uninhabited):
                #if either roof type is not zero, set the prediction to one
                for roof_type in utils.ROOF_TYPES:
                    filtered_detections = self.detections[roof_type][img_name][np.array(self.probs[roof_type][img_name]) > thres]
                    if len(filtered_detections)>0:
                        self.y_pred_any_roof[thres][i] = 1
            self.accuracy_anyRoof.append(classification_report(self.y_true_any_roof, self.y_pred_any_roof[thres]))  


        with open(self.out_path+'class_anyRoof.csv', 'w') as f:
            for thres, score in zip(self.unique_probs,self.accuracy_anyRoof): 
                log = list()
                log.append('{}'.format(thres))
                log.append(score)

                log = '\n'.join(log)
                print log
                f.write(log)


    def get_MAE(self):
        self.errors = dict()
        self.mae = dict()
        for roof_type in utils.ROOF_TYPES:
            self.errors[roof_type] = defaultdict(list)
            self.mae[roof_type] = dict()
            for thres in self.unique_probs:
                for img_name in self.img_names:
                    filtered_detections = self.detections[roof_type][img_name][np.array(self.probs[roof_type][img_name]) > thres]
                    self.errors[roof_type][thres].append(math.fabs(len(self.correct_roofs[roof_type]) - len(filtered_detections)))  
                for img_name in self.uninhabited:
                    filtered_detections = self.detections[roof_type][img_name][np.array(self.probs[roof_type][img_name]) > thres]
                    self.errors[roof_type][thres].append(math.fabs(len(self.correct_roofs[roof_type]) - len(filtered_detections)))
                self.mae[roof_type][thres] = np.mean(np.array(self.errors[roof_type][thres]))
        for roof_type  in utils.ROOF_TYPES:
            with open(self.out_path+'mae_{}.csv'.format(roof_type), 'w') as f:
                for thres in self.unique_probs:
                    f.write('{},{}\n'.format(thres, self.mae[roof_type][thres]))      



          

if __name__=='__main__':
    class_path = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/pipe/with_original_testing_set/viola9999_metalGroup0.1_thatchGroup0.2/class.pickle'
    auc_path = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/pipe/with_original_testing_set/viola9999_metalGroup0.1_thatchGroup0.2/auc.pickle'
    #with open(class_path, 'rb') as f:
    #    classification = pickle.load(f)
    with open(auc_path, 'rb') as f:
        auc = pickle.load(f)
    #auc.


