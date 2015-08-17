import utils



class AucCurve(object):
    def __init__(self, img_names, correct_roofs):
        self.img_names = img_names
        self.correct_roofs
        self.probs = dict()
        self.detections = dict()
        for roof in utils.ROOF_TYPES:
            self.probs[roof] = dict()
            self.detections[roof] = dict()

    def set_detections(self, detections, img_name):
        for roof_type, detects in detections.iteritems():
            self.detections[roof_type][img_name] = detects

    def set_probs(self, probs, img_name): 
        for roof_type, detects in detections.iteritems():
            self.probs[roof_type][img_name] = probs[roof_type]  

    def calculate_auc_values(self, roof_type):
        #find the unique prob points
        #score the entire image set at that prob point
        for thres in unique_probs:
            eval = Evaluation(correct_roofs=self.correct_roofs, detections=self.detections) 
            #find true pos
            #find false pos
        #you get a single number for true pos and false pos at each threshold 
        return auc

    def plot_auc(self):
        self.calculate_auc_values()
        #plot using the values

