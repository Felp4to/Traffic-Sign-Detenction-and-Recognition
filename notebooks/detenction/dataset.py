import yaml
import pandas as pd
import dir
import os
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import random
from collections import Counter
from tqdm import tqdm

CONFIGURATION_FILE = '../../config/config_dataset2.yaml'

# minimum number of images
min_element_training = None
min_element_test = None
min_element_validation = None

# percentage partitions
percentage_training = None
percentage_test = None
percentage_validation = None

# dictionaries
dict_classes = {}  # superclasses
dict_distribution = {}  # classes distributions
mapping_c_sc = {}  # mapping (class : super-class)
classes = None

# number of classes
num_classes = None


def read_config_file():
    global min_element_training
    global min_element_test
    global min_element_validation
    global percentage_training
    global percentage_test
    global percentage_validation
    global dict_classes
    global dict_distribution
    global mapping_c_sc
    global num_classes
    global classes
    # read file yaml
    with open(CONFIGURATION_FILE, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    # min element
    min_element_training = config_data['min_element_training']
    min_element_test = config_data['min_element_test']
    min_element_validation = config_data['min_element_validation']
    # percentage
    percentage_training = config_data['percentage_training']
    percentage_test = config_data['percentage_test']
    percentage_validation = config_data['percentage_validation']
    classes = config_data['super_classes']['names']
    num_classes = len(classes)
    # initialize dictionaries
    for index, elem in enumerate(classes):
        dict_classes[index] = elem
    for index in range(46):
        mapping_c_sc[index] = config_data['mapping_c_sc'][index]
    for index in range(num_classes):
        dict_distribution[index] = 0
    print_info_dataset()


def print_info_dataset():
    print("min_element_training: ", min_element_training)
    print("min_element_test: ", min_element_test)
    print("min_element_validation: ", min_element_validation)
    print("percentage_training:", percentage_training)
    print("percentage_test:", percentage_test)
    print("percentage_validation:", percentage_validation)
    print("classes:", classes)
    print("num_classes:", num_classes)
    print("dict_classes:", dict_classes)
    print("mapping_c_sc:", mapping_c_sc)
    print("dict_distribution:", dict_distribution)


def distribution_classes():
    global dict_classes
    global dict_distribution
    global mapping_c_sc
    global num_classes

    count = 0
    for index in range(0, num_classes):
        dict_distribution[index] = 0
    # loop through all files in the folder
    for label in os.listdir(dir.destination_folder_labels):
        label_path = os.path.join(dir.destination_folder_labels, label)
        # check if it is a file
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # read the first integer of the line
                    count = count + 1
                    line = line.strip()
                    parts = line.split()
                    if len(parts) >= 1 and parts[0].isdigit():
                        dict_distribution[int(parts[0])] += 1
    print("Number of the istances", count)
    print(dict_distribution)
    print_distribution_histogram()


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


def generate_dataset_mapped():
    print("Dataset mapping...\n")
    count = 0
    # verify that the folder exists
    if not os.path.exists(dir.source_folder_labels):
        print(f"The folder '{dir.source_folder_labels}' not exists.")
    else:
        total_iterations = len(os.listdir(dir.source_folder_labels))
        # scan files in the folders
        with tqdm(total=total_iterations, desc="Copia immagini") as pbar:
            for labelname in os.listdir(dir.source_folder_labels):
                new_content = ""
                if labelname.endswith('.txt'):
                    # open file in read mode and write mode
                    with open(os.path.join(dir.source_folder_labels, labelname), 'r+') as file:
                        lines = file.readlines()
                        # control empty label
                        if not lines:
                            continue
                        # iterate for each row and replace the class id with new class id
                        for line in lines:
                            id = ""
                            i = 0
                            while i < len(line) and line[i].isdigit():
                                id += line[i]
                                i += 1
                            if id:
                                parsed_class_id = mapping_c_sc[int(id)]
                                rest = line[i:]
                                newline = f"{parsed_class_id}{rest}"
                                new_content = new_content + newline
                                # create the path of the new file
                    path_new_label = os.path.join(dir.destination_folder_labels, labelname)
                    source_image = os.path.join(dir.source_folder_images, labelname.replace(".txt", ".jpg"))
                    destination_image = os.path.join(dir.destination_folder_images, labelname.replace(".txt", ".jpg"))
                    # open the new file in read mode and write mode
                    with open(path_new_label, 'w') as new_file:
                        count += 1
                        new_file.write(new_content)
                    shutil.copy(source_image, destination_image)
                    #print(f"New file '{path_new_label}' created and saved in the folder '{dir.destination_folder_labels}'.")
                    #print(f"New file '{destination_image}' created and saved in the folder '{dir.destination_folder_images}'.")
                pbar.update(1)
    print("They were created", count, "new files in", dir.destination_folder_labels)
    print("They were created", count, "new files in", dir.destination_folder_images)


def stratified_sampling():
    print("Stratified sampling...\n")
    dataset_path = Path(dir.destination_folder_labels)
    labels = sorted(dataset_path.rglob("*.txt"))
    indx = [l.stem for l in labels]
    labels_df = pd.DataFrame([], columns=dict_distribution.keys(), index=indx)
    for label in labels:
        lbl_counter = Counter()
        with open(label, 'r') as lf:
            lines = lf.readlines()
        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(l.split(' ')[0])] += 1
        labels_df.loc[label.stem] = lbl_counter
    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`
    # display(labels_df)
    # stratified sampling
    partitions_for_each_class = [[] for _ in range(len(dict_distribution))]
    # ascending sorting of keys based on values
    classes_key_orderd_for_freq = sorted(dict_distribution, key=lambda k: dict_distribution[k])
    filenames_used_for_train = set()
    filenames_used_for_test = set()
    filenames_used_for_valid = set()
    # seed random
    random.seed(42)
    # partition the images that have signs
    with tqdm(total=len(classes_key_orderd_for_freq), desc="Copia immagini") as pbar:
        for class_idx in classes_key_orderd_for_freq:
            df_class_idx = labels_df.loc[labels_df[class_idx] != 0.0][class_idx]
            filenames = set(df_class_idx.index)
            filenames.difference_update(filenames_used_for_train)
            filenames.difference_update(filenames_used_for_test)
            filenames.difference_update(filenames_used_for_valid)
            # convert set to list
            filenames = list(filenames)
            # shuffle
            random.shuffle(filenames)
            # calculate length
            total_length = len(filenames)
            # calculate partial lengths
            training_length = max(min_element_training, int(total_length * percentage_training))
            test_length = max(min_element_test, int(total_length * percentage_test))
            # lunghezza_validation = max(min_elementi_validation, int(total_length * percentuale_validation))
            # split lists
            training_set = filenames[:training_length]
            test_set = filenames[training_length:(training_length + test_length)]
            validation_set = filenames[(training_length + test_length):]
            # update the sets of file used
            filenames_used_for_train.update(training_set)
            filenames_used_for_test.update(test_set)
            filenames_used_for_valid.update(validation_set)
            # add for each array the partitions for each class
            partitions_for_each_class[class_idx].append(training_set)
            partitions_for_each_class[class_idx].append(test_set)
            partitions_for_each_class[class_idx].append(validation_set)
            # Aggiorna la barra di avanzamento
            pbar.update(1)

    return partitions_for_each_class


def prova(partitions_for_each_class):
    total_iterations = sum(len(partitions) for partitions in partitions_for_each_class)
    with tqdm(total=total_iterations, desc="Prova") as pbar:
        for class_id, partitions in enumerate(partitions_for_each_class):
            a = 1 + 1


def create_splitted_dataset(partitions_for_each_class):
    #print("Dataset generation...It will take a few minutes...")
    ext_labels = '.txt'
    ext_images = '.jpg'
    count_images = 0
    count_labels = 0
    count_train = 0
    count_test = 0
    count_valid = 0
    count_images_without_signs = 0
    # Calcola il numero totale di iterazioni in base alle tue partizioni
    total_iterations = len(partitions_for_each_class)
    with tqdm(total=total_iterations, desc="Dataset generation") as pbar2:
        # copio i file txt e le immagini con cartelli nelle cartelle giuste
        for class_id, partitions in enumerate(partitions_for_each_class):
            class_folder = str(class_id) + '/'
            destination_class_path = os.path.join(dir.destination_folder_sampling, class_folder)
            destination_train_path = os.path.join(destination_class_path, 'train/')
            destination_test_path = os.path.join(destination_class_path, 'test/')
            destination_valid_path = os.path.join(destination_class_path, 'valid/')
            destination_train_images = os.path.join(destination_train_path, 'images/')
            destination_train_labels = os.path.join(destination_train_path, 'labels/')
            destination_test_images = os.path.join(destination_test_path, 'images/')
            destination_test_labels = os.path.join(destination_test_path, 'labels/')
            destination_valid_images = os.path.join(destination_valid_path, 'images/')
            destination_valid_labels = os.path.join(destination_valid_path, 'labels/')

            # train
            for filename in partitions[0]:
                # images
                # aaa.add(filename)
                file = filename + ext_images
                source_path_image = os.path.join(dir.destination_folder_images, str(file))
                destination_path_image = os.path.join(destination_train_images, str(file))
                shutil.copy(source_path_image, destination_path_image)
                shutil.copy(source_path_image, destination_train_images)
                shutil.copy(source_path_image, dir.destination_folder_sampling_train_images)
                #print("Copy file ", source_path_image, "in to", destination_path_image)
                count_images += 1
                count_train += 1
                # labels
                file = filename + ext_labels
                source_path_label = os.path.join(dir.destination_folder_labels, str(file))
                destination_path_label = os.path.join(destination_train_labels, str(file))
                shutil.copy(source_path_label, destination_path_label)
                shutil.copy(source_path_label, destination_train_labels)
                shutil.copy(source_path_label, dir.destination_folder_sampling_train_labels)
                #print("Copy file ", source_path_label, "in to", destination_path_label)
                count_labels += 1

            # test
            for filename in partitions[1]:
                # images
                # aaa.add(filename)
                file = filename + ext_images
                source_path_image = os.path.join(dir.destination_folder_images, str(file))
                destination_path_image = os.path.join(destination_test_images, str(file))
                shutil.copy(source_path_image, destination_path_image)
                shutil.copy(source_path_image, destination_test_images)
                shutil.copy(source_path_image, dir.destination_folder_sampling_test_images)
                #print("Copy file ", source_path_image, "in to", destination_path_image)
                count_images += 1
                count_test += 1
                # labels
                file = filename + ext_labels
                source_path_label = os.path.join(dir.destination_folder_labels, str(file))
                destination_path_label = os.path.join(destination_test_labels, str(file))
                shutil.copy(source_path_label, destination_path_label)
                shutil.copy(source_path_label, destination_test_labels)
                shutil.copy(source_path_label, dir.destination_folder_sampling_test_labels)
                #print("Copy file ", source_path_label, "in to", destination_path_label)
                count_labels += 1

            # valid
            for filename in partitions[2]:
                # images
                # aaa.add(filename)
                file = filename + ext_images
                source_path_image = os.path.join(dir.destination_folder_images, str(file))
                destination_path_image = os.path.join(destination_valid_images, str(file))
                shutil.copy(source_path_image, destination_path_image)
                shutil.copy(source_path_image, destination_valid_images)
                shutil.copy(source_path_image, dir.destination_folder_sampling_valid_images)
                #print("Copy file ", source_path_image, "in to", destination_path_image)
                count_images += 1
                count_valid += 1
                # labels
                file = filename + ext_labels
                source_path_label = os.path.join(dir.destination_folder_labels, str(file))
                destination_path_label = os.path.join(destination_valid_labels, str(file))
                shutil.copy(source_path_label, destination_path_label)
                shutil.copy(source_path_label, destination_valid_labels)
                shutil.copy(source_path_label, dir.destination_folder_sampling_valid_labels)
                #print("Copy file ", source_path_label, "in to", destination_path_label)
                count_labels += 1
            # Aggiorna la barra di avanzamento
            pbar2.update(1)

    # output
    print(count_images, "images were copied")
    print(count_labels, "labels were copied")
    print(count_train, "training files were copied")
    print(count_test, "test files were copied")
    print(count_valid, "validation files were copied")
