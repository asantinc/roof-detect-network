            THIS IS ALL DONE BY AUC
            #SCORING THE CURRENT IMAGE
            fast_scoring = True# False
            #if self.full_dataset:
            #    fast_scoring = True           
            for t, thres in enumerate(self.auc_thresholds):
                for roof_type in utils.ROOF_TYPES:
                    detections = classified_detections[roof_type][t]
                    print 'NEURAL SCORING FOR THRESHOLD {}'.format(thres)
                    self.detections_after_neural[t].set_detections(img_name=img_name, 
                                                            roof_type=roof_type, 
                                                            detection_list=detections)
                self.evaluation_after_neural[t].score_img(img_name, img_shape[:2], fast_scoring=fast_scoring, contours=self.groupBounds)
            #FINAL EVALUATION
            if self.method == 'viola': 
                if self.pickle_viola is None:
                    self.viola.evaluation.print_report(print_header=True, stage='viola')
                else:
                    self.viola_evaluation.print_report(print_header=True, stage='viola')
            for t, thres in enumerate(self.auc_thresholds):
                self.evaluation_after_neural[t].detections.total_time = neural_time+viola_time
                header = False if self.method=='viola' else True
                self.evaluation_after_neural[t].print_report(print_header=header, stage='neural', report_name='report_thres{}.txt'.format(thres))


