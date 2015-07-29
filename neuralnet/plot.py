import pdb
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils

def plot_loss():
    path = utils.get_path(neural=True, in_or_out=utils.OUT, data_fold=utils.TRAINING)
    for file in os.listdir(path):
        if file.endswith('_history'):
            training_loss = list()
            validation_loss = list()
            with open(path+file, 'rb') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                for row in csv_reader:
                    training_loss.append(float(row[1]))
                    validation_loss.append(float(row[2]))

            plt.plot(training_loss, label='train loss')
            plt.plot(validation_loss, label='valid loss')
            plt.title('History of {0}'.format(file[:-(len('_history'))]))
            plt.legend(loc='best')

            plot_name = path+file+'.jpg' 
            plt.savefig(plot_name)
            plt.close()


if __name__ == "__main__":
    plot_loss()

