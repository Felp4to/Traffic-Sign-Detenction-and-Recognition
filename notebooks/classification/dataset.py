import csv
import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

import dir
import matplotlib.pyplot as plt
import random

CONFIGURATION_FILE = '../../config/config_dataset_classification.yaml'

# percentage partitions
percentage_training = None
percentage_test = None

# dictionaries
dict_classes = {}                                           # superclasses
dict_distribution = {}                                      # classes distributions
mapping_c_sc = {}                                           # mapping (class : super-class)
classes = None
num_classes = None                                          # number of classes
instances = None                                            # array di 43 elementi, ogni elemento è una lista di file

def read_config_file():
    global dict_classes
    global dict_distribution
    global mapping_c_sc
    global num_classes
    global classes
    global instances
    global percentage_training
    global percentage_test
    # read file yaml
    with open(CONFIGURATION_FILE, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    classes = config_data['classes']['names']
    num_classes = len(classes)
    # percentage
    percentage_training = config_data['percentage_training']
    percentage_test = config_data['percentage_test']
    # pulisco array
    instances = [None] * (num_classes - 1)
    instances = [list() for _ in range(num_classes)]
    # initialize dictionaries
    for index, elem in enumerate(classes):
        dict_classes[index] = elem
    for index in range(num_classes):
        dict_distribution[index] = 0
    instances = [list() for _ in range(num_classes)]                # [None] * (num_classes-1)
    print_info_dataset()

def print_info_dataset():
    print("classes:", classes, "\n")
    print("num_classes:", num_classes, "\n")
    print("dict_classes:", dict_classes, "\n")
    #print("dict_distribution:", dict_distribution, "\n")
    #print("instances:", instances, "\n")
    print("percentage_training:", percentage_training, "\n")
    print("percentage_test:", percentage_test, "\n")

def distribution_classes():
    global instances
    count = 0
    # csv file pathes
    csv_train = dir.source_folder + "/train.csv"
    csv_test = dir.source_folder + "/test.csv"
    # read file csv train
    with open(csv_train, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        first = True
        for riga in csv_reader:
            if first:
                # jump header row
                first = False
                continue
            dict_distribution[int(riga["ClassId"])] += 1
            # instances[int(riga["ClassId"])].append(riga["Path"])
            count += 1
    # read file csv test
    with open(csv_test, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        first = True
        for riga in csv_reader:
            if first:
                # jump header row
                first = False
                continue
            dict_distribution[int(riga["ClassId"])] += 1
            instances[int(riga["ClassId"])].append(riga["Path"])
            count += 1
    print("Numer of instances: ", str(count))
    print("Class Distribution: ", dict_distribution)
    print("Instances:", instances)

def print_distribution_histogram():
    # Extract date from the dictionary
    keys = list(dict_distribution.keys())
    values = list(dict_distribution.values())
    # create histogram
    plt.bar(keys, values)
    # add labels and titles
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Histogram of the distribution')
    # show graph
    plt.show()


def create_folders():
    destination_classification = dir.destination_folder + '/Traffic Sign Classification'
    os.makedirs(destination_classification, exist_ok=True)
    destination_test_folder = destination_classification + '/test'
    destination_train_folder = destination_classification + '/train'
    os.makedirs(destination_test_folder, exist_ok=True)
    os.makedirs(destination_train_folder, exist_ok=True)
    # directory delle classi
    for i in range(0, num_classes):
        path = Path(destination_test_folder + '/' + str(i))
        path.mkdir(parents=True, exist_ok=True)

def generate_dataset_for_yolo():
    source_train_folder = dir.source_folder + '/Train'
    destination_test_folder = dir.destination_classification + '/test'
    count = 0
    create_folders()
    with tqdm(total=num_classes, desc="Training set generation") as pbar:
        for elem in range(0, num_classes):
            source = source_train_folder + '/' + str(elem)
            destination = dir.destination_classification + '/train/' + str(elem)
            shutil.copytree(source, destination)
            # update bar
            pbar.update(1)
    with tqdm(total=num_classes, desc="Test set generation") as pbar:
        for class_id in range(0, num_classes):
            for elem in instances[class_id]:
                source = dir.source_folder + '/' + elem
                destination = destination_test_folder + "/" + str(class_id)
                shutil.copy(source, destination)
                count += 1
            # update bar
            pbar.update(1)


