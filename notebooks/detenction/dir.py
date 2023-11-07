import yaml
import os
import dataset
from tqdm import tqdm

# pathes
CONFIGURATION_FILE = "../../config/config_dir.yaml"
source_folder = None
source_folder_images = None
source_folder_labels = None

destination_folder = None
destination_folder_images = None
destination_folder_labels = None
destination_folder_sampling = None
destination_folder_train = None
destination_folder_test = None
destination_folder_valid = None
destination_folder_train_images = None
destination_folder_train_labels = None
destination_folder_test_images = None
destination_folder_test_labels = None
destination_folder_valid_images = None
destination_folder_valid_labels = None


def read_config_file():
    global source_folder
    global destination_folder
    global source_folder_images
    global source_folder_labels
    global destination_folder_images
    global destination_folder_labels
    global destination_folder_sampling
    global destination_folder_train
    global destination_folder_test
    global destination_folder_valid
    global destination_folder_train_images
    global destination_folder_train_labels
    global destination_folder_test_images
    global destination_folder_test_labels
    global destination_folder_valid_images
    global destination_folder_valid_labels
    # read yaml file
    with open(CONFIGURATION_FILE, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    # initialize pathes
    source_folder = config_data['source_folder']
    destination_folder = config_data['destination_folder'] + '/Traffic Sign Detenction'
    source_folder_images = source_folder + '/images'
    source_folder_labels = source_folder + '/labels'
    destination_folder_images = destination_folder + '/images'
    destination_folder_labels = destination_folder + '/labels'
    destination_folder_sampling = destination_folder + '/stratified_sampling/'
    destination_folder_train = destination_folder + '/train'
    destination_folder_test = destination_folder + '/test'
    destination_folder_valid = destination_folder + '/valid'
    destination_folder_train_images = destination_folder_train + '/images'
    destination_folder_train_labels = destination_folder_train + '/labels'
    destination_folder_test_images = destination_folder_test + '/images'
    destination_folder_test_labels = destination_folder_test + '/labels'
    destination_folder_valid_images = destination_folder_valid + '/images'
    destination_folder_valid_labels = destination_folder_valid + '/labels'
    print_info_directories()


def print_info_directories():
    print("source_folder: ", source_folder)
    print("destination_folder: ", destination_folder)
    print("source_folder_images: ", source_folder_images)
    print("source_folder_labels: ", source_folder_labels)
    print("destination_folder_images: ", destination_folder_images)
    print("destination_folder_labels: ", destination_folder_labels)
    print("destination_folder_sampling: ", destination_folder_sampling)


def create_folders():
    os.makedirs(destination_folder_labels)
    os.makedirs(destination_folder_images)
    os.makedirs(destination_folder_sampling)
    # create the folders train, test and valid
    os.makedirs(destination_folder_train)
    os.makedirs(destination_folder_test)
    os.makedirs(destination_folder_valid)
    # create the folders images and labels
    os.makedirs(destination_folder_train_images)
    os.makedirs(destination_folder_train_labels)
    os.makedirs(destination_folder_test_images)
    os.makedirs(destination_folder_test_labels)
    os.makedirs(destination_folder_valid_images)
    os.makedirs(destination_folder_valid_labels)
    for index in range(int(dataset.num_classes)):
        path_class = destination_folder_sampling + '/' + str(index)
        path_class_train = path_class + '/train'
        path_class_train_images = path_class_train + '/images'
        path_class_train_labels = path_class_train + '/labels'
        path_class_test = path_class + '/test'
        path_class_test_images = path_class_test + '/images'
        path_class_test_labels = path_class_test + '/labels'
        path_class_valid = path_class + '/valid'
        path_class_valid_images = path_class_valid + '/images'
        path_class_valid_labels = path_class_valid + '/labels'
        os.makedirs(path_class)
        os.makedirs(path_class_train)
        os.makedirs(path_class_train_images)
        os.makedirs(path_class_train_labels)
        os.makedirs(path_class_test)
        os.makedirs(path_class_test_images)
        os.makedirs(path_class_test_labels)
        os.makedirs(path_class_valid)
        os.makedirs(path_class_valid_images)
        os.makedirs(path_class_valid_labels)



