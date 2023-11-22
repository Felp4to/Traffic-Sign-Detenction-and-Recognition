import os
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT_FOLDER = 'C:/Users/Paolo/Desktop/Prova2/Traffic Sign Detenction/Traffic Sign Detenction/test'
IMAGES_FOLDER = ROOT_FOLDER + '/images'
LABELS_FOLDER = ROOT_FOLDER + '/labels'
CLASSIFY_PATH = '../../models/classification/traint_43_classes_split_70_30/weights/best.pt'
DETENCT_PATH = '../../models/detenction/train_7_classes_split_70_15_15/weights/best.pt'
CONFIGURATION_FILE = '../../config/config_classes.yaml'

model_classify = None
model_detenct = None
num_classes = None
classes = None
filenames = None

def read_config_file():
    global num_classes
    global classes
    # read file yaml
    with open(CONFIGURATION_FILE, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    classes = config_data['classes']['names']
    num_classes = len(classes)


def load_models():
    global model_detenct
    global model_classify
    try:
        model_classify = YOLO(CLASSIFY_PATH)
        model_detenct = YOLO(DETENCT_PATH)
    except FileNotFoundError:
        print(f"Files not found.")
        return []

def print_filenames():
    print("length: ", len(filenames))
    print(filenames)

def init_list():
    global filenames
    try:
        names = os.listdir(IMAGES_FOLDER)
        filenames = [os.path.splitext(file)[0] for file in names if os.path.isfile(os.path.join(IMAGES_FOLDER, file))]
        return filenames
    except FileNotFoundError:
        print(f"Folder '{IMAGES_FOLDER}' not found.")
        return []

# funzione per croppare una porzione di una immagine, ritorna l'immagine croppata
def crop_image(file, coordinates):
    image = Image.open(file)
    crop = image.crop(coordinates)
    #crop.show()
    return crop

def predict(file):
    read_config_file()
    results = model_detenct(file)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmenation masks outputs
        probs = result.probs  # Class probabilities
        # Iterations on each prediction
        for box in result.boxes:
            superclass_id = int(box.cls[0].item())
            superclass_name = result.names[box.cls[0].item()]
            cords = [round(x) for x in box.xyxy[0].tolist()]
            conf_detect = round(box.conf[0].item(), 2)
            # crop.show()
            results2 = model_classify(crop_image(file, cords))
            key = results2[0].probs.top1
            conf_classify = results2[0].probs.top1conf
            conf_classify = conf_classify.item()
            class_id = results2[0].names[key]
            classname = classes[int(class_id)]
            print("-------- Prediction ----------")
            print("SuperClass ID: ", superclass_id)
            print("SuperClass Name:", superclass_name)
            print("Coordinates:", cords)
            print("Confidence (detenct):", conf_detect)
            print("Class ID:", class_id)
            print("Class Name:", classname)
            print("Confidence (classify):", conf_classify)
            print("------------------------------")
            # return superclass_id, superclass_name, cords, conf_detect, class_id,classname, conf_classify

#def funzione_che_mi_dice_se_due_bounding_box_si_riferiscono_allo_stesso_oggetto():
    #
    #
    #