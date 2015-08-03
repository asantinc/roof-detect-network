import utils
import pdb

path = utils.get_path(params=True, neural=True)

log = dict()
log['log'] = 1
log['non_roofs'] = 2
log['viola_data'] ='combo11_min_neighbors3_scale1.08_groupFalse_downsizedFalse_removeOffTrue_mergeFalsePosFalse_vocGood0.3_rotateTrue_separateDetectionsTrue'

file_num = 0
for roof_type in ['metal', 'thatch', 'Both']:
    log['roof_type'] = roof_type
    for layers in [1,2,3,4,5]:
        log['num_layers'] = layers
        for flip in [0,1]:
            log['flip'] = flip
            if layers==5:
                for dropout in [0,1]:
                    log['dropout'] = dropout
                    file_name = path+'params'+str(file_num)+'.csv'
                    with open(file_name, 'w') as f:
                        for k, v in log.iteritems():
                            f.write('{},{}\n'.format(k, v))
                    file_num+=1
            else:
                log['dropout'] = 0 
                fname = path+'params'+str(file_num)+'.csv'
                with open(fname, 'w') as f:
                    for k, v in log.iteritems():
                        f.write('{},{}\n'.format(k, v))
                file_num+=1

