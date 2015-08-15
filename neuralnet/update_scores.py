import csv
from optparse import OptionParser
from collections import defaultdict
from operator import itemgetter
import os
import pdb
import utils


def process_viola_reports(viola_paths, fold=None, title=''):
    all_reports = dict()
    for path, folder_name in viola_paths:
        with open(path+folder_name+'/report{}.txt'.format(title), 'rb') as csvfile:
            all_reports[folder_name] = defaultdict(list)
            reader = csv.reader(csvfile, delimiter='\t')
            all_lines = [line for line in reader]

            if len(all_lines) < 9:
                print 'Failed '+folder_name
                continue

            all_reports[folder_name]['metal'] = dict()
            all_reports[folder_name]['thatch'] = dict()

            all_reports[folder_name]['metal']['detectors'] = all_lines[0][1:]
            all_reports[folder_name]['thatch']['detectors'] = all_lines[1][1:]
            try:
                all_reports[folder_name]['metal']['time'] = float(all_lines[2][2])
                all_reports[folder_name]['thatch']['time'] = float(all_lines[2][2])
            except:
                continue
           
            for roof_type in ['metal', 'thatch']:
                if roof_type == 'metal':
                    metal_scores = all_lines[5]
                elif roof_type == 'thatch':
                    metal_scores = all_lines[9]
                all_reports[folder_name][roof_type]['roofs'] = int(float(metal_scores[0]))
                all_reports[folder_name][roof_type]['detections'] = int(float(metal_scores[2]))
                all_reports[folder_name][roof_type]['Recall'] = float(metal_scores[4])
                all_reports[folder_name][roof_type]['Precision'] = float(metal_scores[6])
                all_reports[folder_name][roof_type]['F1'] = float(metal_scores[8])

    #put the tuples in a list, sort them, write to file
    sorted_tuples = dict()
    log_to_file = list()
    for roof_type in ['metal', 'thatch']:
        log_to_file.append('***************'+roof_type.upper()+'*******************')
        sorted_tuples[roof_type] = defaultdict(list)
        for sort_score in ['Recall', 'F1']:

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
                recall = all_reports[folder_name][roof_type]['Recall']
                precision = all_reports[folder_name][roof_type]['Precision']
                f1 = all_reports[folder_name][roof_type]['F1']
                time = all_reports[folder_name][roof_type]['time']
                log_to_file.append('{}\t{}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.2f}\t{}'.format(roofs, detections, 
                                                                        recall, precision, f1, time, folder_name)) 
            log_to_file.append('\n')

    with open(utils.time_stamped_file('ranking_original.txt'), 'w') as ranking_file:
        ranking_file.write('\n'.join(log_to_file)) 
    print '\n'.join(log_to_file)



def main():
    paths = defaultdict(list)
    #for fold in [utils.TRAINING, utils.TESTING, utils.VALIDATION]:
    for full_dataset in [True, False]:
        for fold in [utils.VALIDATION]:
            viola_path = utils.get_path(in_or_out=utils.OUT, viola=True, data_fold=fold) 
            pipe_path = utils.get_path(in_or_out=utils.OUT, pipe=True, data_fold=fold)

            #for path in [viola_path, pipe_path]:
            for path in [viola_path]:
                for folder in os.listdir(path):
                    if os.path.isfile(path+folder+'/report.txt'):
                        paths[fold].append((path, folder))
            #print viola_paths
            process_viola_reports(paths[fold], fold=fold)


    #same but with the easy reports
    print '\n\n\n'
    paths = defaultdict(list)
    for full_dataset in [True, False]:
        for fold in [utils.VALIDATION]:
            viola_path = utils.get_path(in_or_out=utils.OUT, viola=True, data_fold=fold) 
            pipe_path = utils.get_path(in_or_out=utils.OUT, pipe=True, data_fold=fold)
            sliding_path = utils.get_path(in_or_out=utils.OUT, slide=True, data_fold=fold)

            #for path in [viola_path, pipe_path]:
            for path in [viola_path, sliding]:
                for folder in os.listdir(path):
                    if os.path.isfile(path+folder+'/report_easy.txt'):
                        paths[fold].append((path, folder))

            #print viola_paths
            process_viola_reports(paths[fold], fold=fold, title='_easy')


if __name__ == '__main__':
    main()
