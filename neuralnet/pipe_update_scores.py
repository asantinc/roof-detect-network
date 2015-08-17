import csv
from optparse import OptionParser
from collections import defaultdict
from operator import itemgetter
import os
import pdb

import utils

def process_reports(paths, fold=None, title=''):
    all_reports = dict()
    for path, folder_name in paths:
        with open(path+folder_name+'/report_easy.txt', 'rb') as csvfile:
            all_reports[folder_name] = dict()
            reader = csv.reader(csvfile, delimiter='\t')
            all_lines = [l for l in reader]
            
            for line in all_lines[1:]:
                roof_type = line[1]
                all_reports[folder_name][roof_type] = dict()
                all_reports[folder_name][roof_type]['roofs'] = int(float(line[2])) 
                all_reports[folder_name][roof_type]['time'] = int(float(line[3])) 
                all_reports[folder_name][roof_type]['detections'] = int(float(line[4])) 
                all_reports[folder_name][roof_type]['recall'] = (float(line[5])) 
                all_reports[folder_name][roof_type]['precision'] = float(line[6]) 
                all_reports[folder_name][roof_type]['F1'] = float(line[7]) 

    #put the tuples in a list, sort them, write to file
    sorted_tuples = dict()
    log_to_file = list()
    for roof_type in ['metal', 'thatch']:
        log_to_file.append('***************'+roof_type.upper()+'*******************')
        sorted_tuples[roof_type] = defaultdict(list)
        for sort_score in ['recall', 'F1']:

            log_to_file.append('Ranked by: '+sort_score.upper())
            log_to_file.append('Roofs\tDetections\tRecall\tPrecision\tF1_score\tTotal time\tFolder Name')
            for folder_name in all_reports.keys():
                try:
                    score_to_add = all_reports[folder_name][roof_type][sort_score]
                except:
                    print 'couldnt process {0}'.format(folder_name)
                else:
                    sorted_tuples[roof_type][sort_score].append((folder_name, score_to_add))

            sorted_tuples[roof_type][sort_score] = sorted(sorted_tuples[roof_type][sort_score], key=itemgetter(1), reverse=True)

            #write the sorted thing to file?
            for folder_name, score in sorted_tuples[roof_type][sort_score]:     
                roofs = all_reports[folder_name][roof_type]['roofs']
                detections = all_reports[folder_name][roof_type]['detections']
                recall = all_reports[folder_name][roof_type]['recall']
                precision = all_reports[folder_name][roof_type]['precision']
                f1 = all_reports[folder_name][roof_type]['F1']
                time = all_reports[folder_name][roof_type]['time']
                log_to_file.append('{}\t{}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.2f}\t{}'.format(roofs, detections, 
                                                                        recall, precision, f1, time, folder_name)) 
            log_to_file.append('\n')

    with open(utils.time_stamped_file('ranking'+title), 'w') as ranking_file:
        ranking_file.write('\n'.join(log_to_file)) 
    print '\n'.join(log_to_file)



def main():
    #same but with the easy reports
    print '\n\n\n'
    paths = defaultdict(list)
    for full_dataset in [True]:
        for fold in [utils.VALIDATION]:
            #viola_path = utils.get_path(full_dataset=full_dataset, in_or_out=utils.OUT, viola=True, data_fold=fold) 
            pipe_path = utils.get_path(full_dataset=full_dataset, in_or_out=utils.OUT, pipe=True, data_fold=fold)
            #sliding_path = utils.get_path(full_dataset=full_dataset, in_or_out=utils.OUT, slide=True, data_fold=fold)
            print pipe_path

            #for path in [viola_path, pipe_path]:
            for path in [pipe_path]:#sliding_path]:
                for folder in os.listdir(path):
                    print folder
                    if os.path.isfile(path+folder+'/report_easy.txt'):
                        print 'FOUND'
                        paths[fold].append((path, folder))

            #print viola_paths
            process_reports(paths[fold], fold=fold, title='_easyPipe')



if __name__ == '__main__':
    main()

