import pickle
import os
import pdb

import experiment_settings as settings

def getF1(recall, prec):
    return 2.0*(recall*prec)/(recall+prec)

def write_to_file(roof_type, cov_type, rank_file, metric, ranked_list):
    with open(rank_file, 'a') as f:
        f.write('**************'+metric+'\t'+roof_type+'\t'+cov_type+' **************\n')
        to_print = [str(r[0])+'\t'+str(r[1]) for r in ranked_list] 
        f.write('\n'.join(to_print))
        f.write('\n')


if __name__ == '__main__':
    '''
    Rank each detector combo according to recall and precision
    '''
    #get the filenames in ../output/viola/with_training_set/ 
    results = dict()
    for t in ['testing', 'training']:
        dir = '../output/viola/with_'+t+'_set/'
        #get the results for each 
        for directory in os.listdir(dir):
            if directory.startswith('combo'):
                #get the pickle file with stats
                with open(dir+directory+'/evaluation.pickle', 'rb') as eval:
                    results[directory[:6]] = pickle.load(eval)

        #ranking file
        rank_f_name = dir+'rank.csv'
        open(rank_f_name, 'w').close()
        threshold = 2     #value for which we want to find the ranking between detectors

        ranked_list = dict()
        for roof_type in ['metal', 'thatch']:
            ranked_list[roof_type] = dict()
            for cov_type in ['any', 'single']:
                ranked_list[roof_type][cov_type] = dict()
                for metric in ['recall','precision']:
                    score_tups = [(name, results[name][roof_type][cov_type][metric][threshold]) for name in results.keys() if results[name][roof_type] != False]   
                    #reorder it as needed         
                    ranked_list[roof_type][cov_type][metric] = sorted(score_tups, key=lambda tup: tup[1], reverse=True)
                    write_to_file(roof_type, cov_type, rank_f_name, metric, ranked_list[roof_type][cov_type][metric])            

                    
        

