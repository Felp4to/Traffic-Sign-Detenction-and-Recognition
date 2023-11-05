import yaml
import os
import dataset
from tqdm import tqdm

# pathes
CONFIGURATION_FILE = "../../config/config_dir_classification.yaml"

source_folder = None
destination_folder = None

def read_config_file():
    global source_folder
    global destination_folder
    # read yaml file
    with open(CONFIGURATION_FILE, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    # initialize patches
    source_folder = config_data['source_folder']
    destination_folder = config_data['destination_folder']
    print_info_directories()


def print_info_directories():
    print("source_folder: ", source_folder)
    print("destination_folder: ", destination_folder)