import utils
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from reporting import Detections, Evaluation
from scipy.integrate import simps
import pdb

class AucCurve(object):
    def __init__(self, img_names, correct_roofs, out_path, method):
        self.img_names = img_names
        self.correct_roofs = correct_roofs
        self.out_path = out_path
        self.probs = dict()
        self.detections = dict()
        self.method = method

        self.true_pos = defaultdict(list)
        self.false_pos = defaultdict(list)

        for roof in utils.ROOF_TYPES:
            self.probs[roof] = dict()
            self.detections[roof] = dict()


    def set_detections(self, detections, img_name):
        for roof_type, detects in detections.iteritems():
            self.detections[roof_type][img_name] = np.array(detects)

    def set_probs(self, probs, img_name): 
        #TODO: update the probs per image
        for roof_type, detects in probs.iteritems():
            self.probs[roof_type][img_name] = np.array(probs[roof_type])  


    def calculate_auc_values(self):
        detections = dict()
        probs = dict()
        self.recall = defaultdict(list)
        self.precision = defaultdict(list)
        self.easy_recall = defaultdict(list)
        self.easy_precision = defaultdict(list)
        self.area = dict()
        self.easy_area = dict()

        #TODO: find the unique prob points
        unique_probs = [0.0,0.1,.2,.3,.4,.5,.6,.7,.8,.9]
        unique_probs = set()
        unique_probs.update([1.])
        for roof_type in utils.ROOF_TYPES:
            for img_name in self.img_names:
                probs = [int(100*p) for p in list(self.probs[roof_type][img_name])]
                probs = [(float(p)/100) for p in probs]
                unique_probs.update((probs))
        print unique_probs

        unique_probs = sorted(unique_probs)
        for thres in unique_probs:
            print 'TRESHOLD: {}'.format(thres)
            #find true pos
            #for current threshold, for each roof type separately
            eval = Evaluation(correct_roofs=self.correct_roofs, img_names=self.img_names, method=self.method) 
            eval.detections = Detections()
            for roof_type in utils.ROOF_TYPES:
                for img_name in self.img_names:
                    eval.detections.update_roof_num(self.correct_roofs[roof_type][img_name], roof_type)

            filtered_detections = dict()
            total_detections = defaultdict(int)#detections for current threshols, across all images, but per roof_type 
            for img_name in self.img_names:
                #score the entire image set at that prob point
                filtered_detections[img_name] = dict()
                for roof_type in utils.ROOF_TYPES:
                    #the detections are the viola detections
                    #the filtering is done with the probs taken from the neural network
                    filtered_detections[img_name][roof_type] = self.detections[roof_type][img_name][np.array(self.probs[roof_type][img_name]) > thres]
                    total_detections[roof_type] += len(filtered_detections[img_name][roof_type])
                    eval.detections.set_detections(detection_list=filtered_detections[img_name][roof_type], img_name=img_name, roof_type=roof_type)

                 
                eval.score_img(img_name, (1200,2000), fast_scoring=True, write_file=False)

            for roof_type in utils.ROOF_TYPES:
                if total_detections[roof_type]>0 and eval.detections.roof_num[roof_type]>0:
                    recall = float(eval.detections.true_positive_num[roof_type])/(eval.detections.roof_num[roof_type])
                    precision = float(eval.detections.true_positive_num[roof_type])/(total_detections[roof_type]) 
                else:
                    recall = 0
                    precision = 1 
                self.recall[roof_type].append(recall)
                self.precision[roof_type].append(precision)

                if total_detections[roof_type]>0 and eval.detections.roof_num[roof_type]>0:
                    easy_recall = float(eval.detections.easy_true_pos_num[roof_type])/(eval.detections.roof_num[roof_type])
                    easy_precision = float(eval.detections.easy_true_pos_num[roof_type])/(eval.detections.easy_true_pos_num[roof_type]+eval.detections.easy_false_pos_num[roof_type]) 
                else:
                    easy_recall = 0
                    easy_precision = 1
                self.easy_recall[roof_type].append(easy_recall)
                self.easy_precision[roof_type].append(easy_precision)


        for roof_type in utils.ROOF_TYPES:
            self.recall[roof_type] = np.array(self.recall[roof_type])
            self.precision[roof_type] = np.array(self.precision[roof_type])
            self.area[roof_type] = simps(self.recall[roof_type], self.precision[roof_type])

            self.easy_recall[roof_type] = np.array(self.easy_recall[roof_type])
            self.easy_precision[roof_type] = np.array(self.easy_precision[roof_type])
            self.easy_area[roof_type] = simps(self.easy_recall[roof_type], self.easy_precision[roof_type])
        print self.area
        print self.easy_area


    def plot_auc(self):
        self.calculate_auc_values()
        for roof_type in utils.ROOF_TYPES:
            # Plot Precision-Recall curve

            plt.clf()
            plt.plot(self.recall[roof_type], self.precision[roof_type], label='Precision-Recall curve: AUC = {0:0.2f}'.format(self.area[roof_type]))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('Precision-Recall {0}: AUC={1:0.2f}'.format(roof_type, self.area[roof_type]))
            plt.legend(loc="lower left")
            plt.savefig('{0}{1}_auc{2:0.2f}.jpg'.format(self.out_path, roof_type, self.area[roof_type]))

            print roof_type
            print self.recall[roof_type]
            print self.precision[roof_type]
        pdb.set_trace()

        for roof_type in utils.ROOF_TYPES:
            # Plot Precision-Recall curve

            plt.clf()
            plt.plot(self.easy_recall[roof_type], self.easy_precision[roof_type], label='Precision-Recall curve: AUC = {0:0.2f}'.format(self.easy_area[roof_type]))
            plt.xlabel('Easy Recall')
            plt.ylabel('Easy Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            #plt.title('Precision-Recall {0}: AUC={1:0.2f}'.format(roof_type, self.area[roof_type]))
            plt.legend(loc="lower left")
            plt.savefig('{0}{1}_auc{2:0.2f}_easy.jpg'.format(self.out_path, roof_type, self.easy_area[roof_type]))

            print roof_type
            print self.recall[roof_type]
            print self.precision[roof_type]

