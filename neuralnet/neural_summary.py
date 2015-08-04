import pdb
import csv
import os

import utils
import operator

def load_neural_results():
    scores = dict()
    path = utils.get_path(neural=True, in_or_out=utils.OUT, data_fold=utils.TRAINING) 

    for file in os.listdir(path):
        if file.startswith('conv') and not (file.startswith('.') or file.endswith('.jpg') or file.endswith('_history') or file.endswith('_layers')):
            with open(path+file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                all_rows = list()
                for i, row in enumerate(reader):
                    all_rows.append(row)
            if i < 1:
                continue
            else:
                scores[file] = dict()
                for key, value in zip(all_rows[0], all_rows[1]):
                    scores[file][key] = value
                
    return scores


def filter(scores, key, value):
    filtered = list()
    for file, key_value in scores.iteritems():
        if key in key_value:
            if key_value[key] == value:
                filtered.append(file)
    return filtered

def sort(files, key, scores):
    sorted_files = list()
    for f in files:
        if key in scores[f]:
            sorted_files.append((f, scores[f][key]))
        else:
            sorted_files.append((f, -1))
    sorted_files.sort(key=operator.itemgetter(1), reverse=True)
    return sorted_files

def print_scores(sorted_files, scores, columns):
    keys = columns
    #for f in sorted_files:
    #    keys.update(scores[f[0]].keys()) 
    log = list()
    log.append('\t'.join([k for k in keys]))
        
    for f in sorted_files:
        file = f[0]
        current_file = list()
        for k in keys:
            if k in scores[file] and k!= 'net_name':
                value = scores[file][k]
                current_file.append(value)
            elif k !='net_name':
                print k
                pdb.set_trace()
                current_file.append('NaN')
        current_file.append(file)

        log.append('\t'.join(current_file))
    return '\n'.join(log)


if __name__ == '__main__':
    scores = load_neural_results()
    metal_only = filter(scores, 'roof_type', 'metal')   
    sorted_metal = sort(metal_only, 'best_valid_accuracy', scores) 
    columns = ['best_valid_accuracy', 'best_valid_loss', 'best_epoch', 'roof_type', 'layers', 'augment', 'dropout', 'time', 'net_name']
    print print_scores(sorted_metal, scores, columns)

    print 'THATCH'
    thatch_only = filter(scores, 'roof_type', 'thatch')
    sorted_metal = sort(thatch_only, 'best_valid_accuracy', scores) 
    columns = ['best_valid_accuracy', 'best_valid_loss', 'best_epoch', 'roof_type', 'layers', 'augment', 'dropout', 'time', 'net_name']
    print print_scores(sorted_metal, scores, columns)


         


