import cPickle as pickle

from neural_network import DataScaler
from neural_data_setup import NeuralDataLoad
import utils

from sklearn.externals import joblib
import json

if __name__ == '__main__':
    #pickle the scaler so we can reuse it later
    net_name = 'metal'
    path = 'debug/' 
    with open('{0}{1}_scaler.pkl'.format(path, net_name), 'rb') as f:
        scaler = pickle.load(f)

    with open('debug/after_mean.json', 'w') as f:
        mean = scaler.mean_
        json.dump(list(mean), f)


    with open('debug/after_std.json', 'w') as f:
        data = scaler.std_
        json.dump(list(data), f)



    
 
