import cPickle as pickle

from neural_network import DataScaler
from neural_data_setup import NeuralDataLoad
import utils

from sklearn.externals import joblib
import json


if __name__ == '__main__':
    print 'Loading data...\n'
    roof_type= 'metal'
    net_name = roof_type 
    data_folder ='scale1.3_minSize50-50_windowSize15-15_stepSize30_dividedSizes' 
    X, y = NeuralDataLoad(data_path=data_folder).load_data(roof_type='metal', non_roofs=1) 
    print 'Data is loaded \n'
    #set up the data scaler
    scaler = DataScaler()
    X = scaler.fit_transform(X)

    #pickle the scaler so we can reuse it later
    path = 'debug/' 
    with open('{0}{1}_scaler.pkl'.format(path, net_name), 'wb') as f:
        pickle.dump(scaler, f, -1)

    joblib.dump(scaler, '{}{}_joblib.pkl'.format(path, net_name)) 
    
    #scaler = joblib.load('{}{}_joblib.pkl'.format(path, net_name)) 

    #with open('{0}{1}_scaler.pkl'.format(path, net_name), 'rb') as f:
    #    scaler = pickle.load(f)

    with open('debug/before_mean.json', 'w') as f:
        mean = list(scaler.mean_)
        json.dump(mean, f)


    with open('debug/before_std.json', 'w') as f:
        data = list(scaler.std_)
        json.dump(data, f)



    
