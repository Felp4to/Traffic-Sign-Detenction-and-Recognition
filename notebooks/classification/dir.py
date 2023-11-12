import yaml

# pathes
CONFIGURATION_FILE = "../../config/config_dir.yaml"

source_folder = None
destination_folder = None
destination_classification = None


def read_config_file():
    global source_folder
    global destination_folder
    global destination_classification
    # read yaml file
    with open(CONFIGURATION_FILE, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    # initialize patches
    source_folder = config_data['source_folder_classify']
    destination_folder = config_data['destination_folder']
    destination_classification = destination_folder + '/Traffic Sign Classification'
    print_info_directories()


def print_info_directories():
    print("source_folder: ", source_folder)
    print("destination_folder: ", destination_folder)
