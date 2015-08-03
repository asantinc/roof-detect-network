import pandas as pd
import pdb
import os

import utils

def load_neural_results():
    #neural_path = utils.get_path(neural=True, in_or_out=utils.OUT, data_fold=utils.TRAINING) 
    neural_path = './'
    frame = None

    print os.listdir(neural_path)
    for file in os.listdir(neural_path):
        #if file.startswith('conv') and not (file.startswith('.') or file.endswith('.jpg') or file.endswith('_history') or file.endswith('_layers')):
        if file.endswith('.csv'):
            print file
            #get the parameters defined
            #the filename is the key to all entries
            pdb.set_trace()
            f = pd.read_csv(neural_path+file ) 
            if frame is None:
                frame = f
            else:
                frame.append(f)
    return frame


if __name__ == '__main__':
    frame = load_neural_results()
    print frame
    pdb.set_trace()

         




