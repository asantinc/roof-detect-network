    '''
    def print_report(self, img_name=None, detections_num=-1, final_stats=False):
        with open(self.report_file, 'a') as report:
                self.thatch_candidates += len(self.overlap_dict[img_name]["thatch"])
                self.metal_candidates += len(self.overlap_dict[img_name]["metal"])
                        
                #report true positives
                true_metal = true_thatch = 0
                for roof_type in self.overlap_dict[img_name].keys():
                    fully_classified, mostly_classified, partially_classified = self.score_detections(roof_type=roof_type, img_name=img_name)

                    self.overlap_all[roof_type]['any'] += sum(self.overlap_roof_with_detections[img_name][roof_type]['any']) 
                    self.overlap_all[roof_type]['single'] += sum(self.overlap_roof_with_detections[img_name][roof_type]['single']) 

                    print self.overlap_dict[img_name][roof_type]
                    for v in self.overlap_dict[img_name][roof_type]:
                        if v > .20:
                            if roof_type == 'metal':
                                true_metal += 1
                            elif roof_type == 'thatch':
                                true_thatch += 1
                            else:
                                raise ValueError('Roof type {0} does not exist.'.format(roof_type))
                self.tp_metal += true_metal
                self.tp_thatch += true_thatch
                log = 'Image '+img_name+' thatch: '+str(true_thatch)+'/'+str(len(self.overlap_dict[img_name]['thatch']))+'\n'
                log = 'Image '+img_name+' metal: '+str(true_metal)+'/'+str(len(self.overlap_dict[img_name]['metal']))+'\n'
                print log
                report.write(log)
            
            #Print number of false positives, False negatives
            log = ('******************************* RESULTS ********************************* \n'
                +'METAL: \n'+
                +'Precision: \t'+str(self.tp_metal)+'/'+str(self.metal_candidates)+
                +'\n'+'Recall: \t'+str(self.tp_metal)+'/'+str(self.all_true_metal)+'\n'+
                +'THATCH: \n'+
                +'Precision: \t'+str(self.tp_thatch)+'/'+str(self.thatch_candidates)+
                +'\n'+'Recall: \t'+str(self.tp_thatch)+'/'+str(self.all_true_thatch)+'\n')
            print log
            report.write(log)
    '''


