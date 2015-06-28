import pdb
import csv
import experiment_settings as settings
import matplotlib.pyplot as plt

def plot_loss(net_name):
    file_name = settings.OUT_HISTORY+str(net_name)
    training_loss = list()
    validation_loss = list()

    with open(file_name, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            training_loss.append(float(row[1]))
            validation_loss.append(float(row[2]))

    plt.plot(training_loss, label='train loss')
    plt.plot(validation_loss, label='valid loss')
    plt.legend(loc='best')
    plt.savefig(settings.OUT_IMAGES+net_name+'_loss.png')


if __name__ == "__main__":
    plot_loss(net_name='conv1_nonroofs1_test20_roofs')

