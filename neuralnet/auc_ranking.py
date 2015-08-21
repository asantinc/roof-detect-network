import utils
import pdb
import os.path
import os
from collections import defaultdict
import csv

def process_report(general_path, folder, file, record):
    with open(general_path+folder+'/'+file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        items = [item for item in reader]
        #print items
        #pdb.set_trace()
        for item in items:
            roof_type = item[1]
            if roof_type == 'metal' or roof_type == 'thatch':
                record[roof_type][2] = int(float(item[4])/6)#detections
                record[roof_type][3] = float(item[5])#recall
                record[roof_type][4] = float(item[6])#precision
                if item[7].endswith('neural'):
                    item[7] = item[7][:-len('neural')]
                record[roof_type][5] = float(item[7])#F1
                record[roof_type][6] = (float(item[3])/6)#time/image
                #neural  metal   90      263.036600828   697     0.888888888889  0.326530612245  0.477611940299
    return record



if __name__ == '__main__':
    scores = dict()
    general_path = '/afs/inf.ed.ac.uk/group/ANC/s0839470/output/pipe/with_original_validation_set/'
    for folder in os.listdir(general_path):
        print folder
        if os.path.isdir(general_path+folder):
            record = dict()
            record_easy = dict()
            for roof_type in utils.ROOF_TYPES:
                record[roof_type] = [-1]*7
                record_easy[roof_type] = [-1]*7

            for file in os.listdir(general_path+folder):
                if 'auc' in file:
                    if 'metal' in file:
                        roof_type = 'metal'
                    elif 'thatch' in file:
                        roof_type = 'thatch'
                    else:
                        continue
                    if 'easy' in file:
                        method = 'easy'
                        start = file.find('auc0.')
                        real_start = start+len('auc0.')
                        end = real_start+file[real_start:].find('_')
                        auc = float(file[real_start:end])/100
                        record_easy[roof_type][0] = folder
                        record_easy[roof_type][1] = auc

                    else:
                        method = 'standard'
                        start = file.find('auc0.')
                        real_start = start+len('auc0.')
                        end = real_start+file[real_start:].find('.')
                        auc = float(file[real_start:end])/100
                        record[roof_type][0] = folder
                        record[roof_type][1] = auc

                #print file
                #pdb.set_trace()
                if file == 'report_thres0.5_easy.txt':
                    record_easy = process_report(general_path, folder, file, record_easy)
                if file == 'report_thres0.5.txt':
                    record = process_report(general_path, folder, file, record)
                   
            for roof_type in utils.ROOF_TYPES:
                if roof_type not in scores:
                    scores[roof_type] = defaultdict(list)
                if record_easy[roof_type][0]!=-1:
                    scores[roof_type]['easy'].append(record_easy[roof_type])
                if record[roof_type][0]!=-1:
                    scores[roof_type]['standard'].append(record[roof_type])


    for roof_type in utils.ROOF_TYPES:
        for method in ['easy', 'standard']:
            log = list()
            viola_num = 0
            slide_num = 0
            prev_viola_auc = -1
            prev_slide_auc = -1
            scores[roof_type][method].sort(key=lambda tup:tup[1], reverse=True)
            log.append('{}&{}&{}&{}&{}&{}&{}\\\\'.format('Detector','Group Threshold' , 'Auc', 'Detections', 'Recall','Precision','F1','Time (s.)'))
            for score in scores[roof_type][method]:
                if 'viola' in score[0]:
                    viola_num += 1
                    if viola_num > 10:
                        continue
                    if prev_viola_auc == score[1]:
                        prev_viola_auc = score[1]
                        continue
                    prev_viola_auc = score[1]
                elif 'slide' in score[0]:
                    slide_num += 1
                    if slide_num > 10:
                        continue
                    if prev_slide_auc == score[1]:
                        prev_slide_auc = score[1]
                        continue
                    prev_slide_auc = score[1]
                end_name = score[0].find('_') 
                name = score[0][:end_name]
                group_thres = float(utils.get_param_value_from_filename(score[0], 'group', separator='_'))
                log.append('{0}&\({7}\)&\({1}\)&\({2}\)&\({3:.3f}\)&\({4:.3f}\)&\({5:.3f}\)&\({6:.3f}\)\\\\'.format(name,  score[1], 
                                                                                score[2], score[3], score[4], score[5], score[6], group_thres))
            output = '\n'.join(log)
            with open('results/{}_{}.txt'.format(roof_type, method), 'w') as f:
                f.write(output)
            print output



