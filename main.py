import argparse
import os
import torch.nn as nn
import torch.optim as optim
import time

import dataset
import util
import network
from training.Train_And_Test import Train_And_Test
from util import plotutil

# Config related
config_filepath = 'config/config.json'
config = util.get_config(config_filepath)
base_data_dir = config['base_dir']
total_epoch = config['total_epoch']
instrument_list = config['instrument_list'].split(',')
instrument_count = int(config['instrument_count'])
instrument_src_list = config['instrument_src'].split(',')
instrument_label = config['instrument_label']
sample_rate = int(config['sample_rate'])
filters = config['filters']
kernels = config['kernels']
padding = config['padding']
pooling = config['pooling']
dense = config['dense']
device = util.get_device()

# data related
data_loaders = []

def load_dataset(data_type, pre_process, batch_size):
    for split in ['train', 'test', 'valid']:
        data_dir = os.path.join(base_data_dir, config[split]['audio_dir']) 
        label_dir = os.path.join(base_data_dir, config[split]['label_dir']) 
        
        loader = dataset.load_dataset(data_type, label_dir, data_dir, instrument_label, batch_size, instrument_list, sample_rate, n_mfcc=40)
        data_loaders.append(loader)
    return data_loaders


def train(data_type, batch_size, learning_rate, pre_process):
    print('Main params: batch_size, learning_rate, pre_process -->', batch_size, learning_rate, pre_process)

    batch_size = int(batch_size)
    learning_rate = float(learning_rate)

    data_loaders = load_dataset(data_type, pre_process, batch_size)
    train_loader = data_loaders[0]
    test_loader = data_loaders[1]
    validate_loader = data_loaders[2]

    input_shape = train_loader.dataset[0][0].shape
    num_classes = len(instrument_list)
    model = network.load_network(input_shape, num_classes, filters, kernels, padding, pooling, dense)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    loss = nn.CrossEntropyLoss()
    model.to(device)
    print("Input shape", input_shape)
    print(model)
    
    train_and_test = Train_And_Test(total_epoch, input_shape, model, optimizer, loss, device, train_loader, test_loader, validate_loader)
    totalTime = 0.0
    all_training_loss = []
    all_validation_loss = []
    all_validation_accuracy = []
    for epoch in range(0, total_epoch):
        start = time.perf_counter()
        training_loss = train_and_test.train(epoch)
        all_training_loss.append(training_loss)

        validate_loss, validate_accuracy = train_and_test.validate()
        all_validation_loss.append(validate_loss)
        all_validation_accuracy.append(validate_accuracy)

        timetaken = time.perf_counter() - start
        totalTime = totalTime + timetaken
        print('This epoch time taken - ', timetaken, 'seconds')

    test_loss, test_accuracy = train_and_test.test()

    suffix = util.create_filename_suffix(data_type, 'batch{}'.format(batch_size), instrument_list, instrument_count, learning_rate, ''.join(instrument_src_list))
    model_location = 'models/cnn-nsynth-classification{}.pt'.format(suffix)
    loss_plot_location = 'models/loss_and_accuracy{}.png'.format(suffix)
    cm_plot_location = 'models/confusion_matrix{}.png'.format(suffix)
    
    train_and_test.save_model(model_location)
    cm = train_and_test.build_cm()
    plotutil.plot_losses(all_training_loss, all_validation_loss, all_validation_accuracy, loss_plot_location)
    plotutil.plot_confusion_matrix(cm, instrument_list, filename=cm_plot_location)

    return 'success'

def plot_cm(data_type, batch_size, learning_rate):
    learning_rate = float(learning_rate)

    data_loaders = load_dataset(data_type, 'False', 100)
    train_loader = data_loaders[0]
    test_loader = data_loaders[1]
    validate_loader = data_loaders[2]
    
    input_shape = test_loader.dataset[0][0].shape
    num_classes = len(instrument_list)
    model = network.load_network(input_shape, num_classes, filters, kernels, padding, pooling, dense)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    suffix = util.create_filename_suffix(data_type, 'batch{}'.format(batch_size), instrument_list, instrument_count, learning_rate, ''.join(instrument_src_list))
    model_location = 'models/cnn-nsynth-classification{}.pt'.format(suffix)
    cm_plot_location = 'models/confusion_matrix{}.png'.format(suffix)
    
    train_and_test = Train_And_Test(total_epoch, input_shape, model, optimizer, None, device, train_loader, test_loader, validate_loader)
    model = train_and_test.load_model(model_location, device)
    model.to(device)
    cm = train_and_test.build_cm()
    plotutil.plot_confusion_matrix(cm, instrument_list, filename=cm_plot_location)

# python main.py --operation=train --data_type=mfcc --batch_size=16 --learning_rate=0.0001
# python main.py --operation=train --data_type=if --batch_size=16 --learning_rate=0.0001
# python main.py --operation=train --data_type=mel_harmonics --batch_size=64 --learning_rate=0.0001
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', required=True)
    parser.add_argument('--data_type', required=True)
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--learning_rate', required=True)
    parser.add_argument('--pre_process', required=False)
    
    args = parser.parse_args()
    if args.operation == 'train':
        print("Operation: Train")
        train(data_type=args.data_type, batch_size=args.batch_size, learning_rate=args.learning_rate, pre_process=args.pre_process)
    
    if args.operation == 'cm':
        print("Operation: Confusion Matrix")
        plot_cm(args.data_type, batch_size=args.batch_size, learning_rate=args.learning_rate)