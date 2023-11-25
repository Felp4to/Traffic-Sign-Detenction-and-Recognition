import os
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import time
import random



ROOT_FOLDER = 'C:/Users/Paolo/Desktop/Prova2/Traffic Sign Detenction/Traffic Sign Detenction/test'
IMAGES_FOLDER = ROOT_FOLDER + '/images'
LABELS_FOLDER = ROOT_FOLDER + '/labels'
CLASSIFY_PATH = '../../models/classification/traint_43_classes_split_70_30/weights/best.pt'
DETENCT_PATH = '../../models/detenction/train_7_classes_split_70_15_15/weights/best.pt'
CONFIGURATION_FILE = '../../config/config_classes.yaml'

columns = [
        'image_path', 'filename', 'superclass_id_predict', 'superclass_name_predict', 'cords_predict',
        'cords_annotation', 'class_id_predict', 'class_id_annotation', 'class_name_predict',
        'conf_detenct_predict', 'conf_classify_predict', 'crop_image_annotation', 'crop_image_predict', 'instance_state'
    ]

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

def labeling(file):
    annotations = []
    path = LABELS_FOLDER + '/' + file + '.txt'
    path_image = IMAGES_FOLDER + '/' + file + '.jpg'
    image = Image.open(path_image)
    width, height = image.size
    # prendere le dimensioni dell'immagine
    with open(path, 'r') as file:
        for row in file:
            values = row.strip().split()
            class_id = int(values[0])
            c1 = float(values[1])
            c2 = float(values[2])
            c3 = float(values[3])
            c4 = float(values[4])
            yolo_cords = (c1,c2,c3,c4)
            cords = yolo_to_minmax(yolo_cords, width, height)
            crop_annotation = crop_image(path_image, cords)
            # crop_annotation.show()
            annotation = (class_id, cords, crop_annotation)
            annotations.append(annotation)
    return annotations

def predict(file):
    predictions = []
    read_config_file()
    path = IMAGES_FOLDER + '/' + file + '.jpg'
    results = model_detenct(path, conf=0.05, imgsz=1024, verbose=False)
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
            crop_predict = crop_image(path, cords)
            # crop_predict.show()
            results2 = model_classify(crop_image(path, cords), verbose=False)
            key = results2[0].probs.top1
            conf_classify = results2[0].probs.top1conf
            conf_classify = conf_classify.item()
            class_id = results2[0].names[key]
            classname = classes[int(class_id)]
            prediction = (superclass_id, superclass_name, cords, conf_detect, class_id, classname, conf_classify, crop_predict)
            # print_prediction(prediction)
            predictions.append(prediction)
    return predictions

def generate_dataframe(files):
    df_total = pd.DataFrame(columns=columns)
    total_iterations = len(files)
    with tqdm(total=total_iterations, desc="Prections") as pbar:
        for file in files:
            df = pd.DataFrame(columns=columns)
            df = generate_dataframe_for_file(file)
            df_total = pd.concat([df_total.dropna(axis=1, how='all'), df.dropna(axis=1, how='all')], ignore_index=True)
            pbar.update(1)
    return df_total

def generate_dataframe_for_file(file):
    df = pd.DataFrame(columns=columns)
    index_to_delete_A = []
    index_to_delete_P = []
    image_path = IMAGES_FOLDER + '/' + str(file) + '.jpg'
    filename = file + '.jpg'
    annotations = labeling(file)
    predictions = predict(file)
    #print("Annotations ->", annotations)
    #print("Predictions ->", predictions)
    # loop on the list annotations
    for indexA, annotation in enumerate(annotations):
        class_id_annotation, cords_annotation, crop_annotation = annotation
        for indexP, prediction in enumerate(predictions):
            (superclass_id_predict, superclass_name_predict, cords_predict, conf_detenct_predict, class_id_predict,
             class_name_predict, conf_classify_predict, crop_prediction) = prediction
            #print(match(cords_predict, cords_annotation))
            if(match(cords_predict, cords_annotation) == True):
                class_id_predict = int(class_id_predict)
                # if the prediction is correct
                if(class_id_predict == class_id_annotation):
                    new_row = [image_path, filename, superclass_id_predict, superclass_name_predict, cords_predict,
                               cords_annotation, class_id_predict, class_id_annotation, class_name_predict,
                               conf_detenct_predict, conf_classify_predict, crop_annotation, crop_prediction, 'RIGHT PREDICTION']
                    df.loc[len(df)] = new_row
                else:  # aggiungo riga math OK se ClassID corrispondono
                    new_row = [image_path, filename, superclass_id_predict, superclass_name_predict, cords_predict,
                               cords_annotation, class_id_predict, class_id_annotation, class_name_predict,
                               conf_detenct_predict, conf_classify_predict, crop_annotation, crop_prediction, 'WRONG PREDICTION']
                    df.loc[len(df)] = new_row
                index_to_delete_A.append(indexA)
                index_to_delete_P.append(indexP)
    # delete matches
    annotations = delete_indexes(annotations, index_to_delete_A)
    predictions = delete_indexes(predictions, index_to_delete_P)
    # add lines for annotations
    for index, annotation in enumerate(annotations):
        # add line
        new_row = [image_path, filename, None, None, None,
                   cords_annotation, None, class_id_annotation, None,
                   None, None, crop_annotation, None, 'NOT DETECTED']
        df.loc[len(df)] = new_row
    # add lines for predictions
    for index, prediction in enumerate(predictions):
        # add line
        new_row = [image_path, filename, superclass_id_predict, superclass_name_predict, cords_predict,
                   None, class_id_predict, None, class_name_predict,
                   conf_detenct_predict, conf_classify_predict, None, crop_prediction, 'NO LABELED']
        df.loc[len(df)] = new_row
    return df


def delete_indexes(original_list, index_to_delete):
    result = [elemento for indice, elemento in enumerate(original_list) if indice not in index_to_delete]
    return result


def print_prediction(prediction):
    (superclass_id, superclass_name, cords, conf_detect, class_id, classname, conf_classify) = prediction
    print("-------- Prediction ----------")
    print("SuperClass ID: ", superclass_id)
    print("SuperClass Name:", superclass_name)
    print("Coordinates:", cords)
    print("Confidence (detenct):", conf_detect)
    print("Class ID:", class_id)
    print("Class Name:", classname)
    print("Confidence (classify):", conf_classify)
    print("------------------------------")

def yolo_to_minmax(yolo_annotation, image_width, image_height):
    x_center, y_center, width, height = yolo_annotation
    # Calcolare le coordinate minime e massime
    min_x = max(0, int((x_center - width / 2) * image_width))
    min_y = max(0, int((y_center - height / 2) * image_height))
    max_x = min(int((x_center + width / 2) * image_width), image_width)
    max_y = min(int((y_center + height / 2) * image_height), image_height)
    return min_x, min_y, max_x, max_y

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    # Calcolare le aree delle bounding box
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)
    # Calcolare l'area di intersezione delle bounding box
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    area_overlap = x_overlap * y_overlap
    # Calcolare l'area dell'unione delle bounding box
    area_union = area_box1 + area_box2 - area_overlap
    # Calcolare l'IoU (Intersezione su Unione)
    iou = area_overlap / max(1e-5, area_union)
    return iou

def match(box1, box2):
    iou = calculate_iou(box1, box2)
    # print("iou: ", iou)
    if iou > 0.5:
        return True
    return False

def metrics(df):
    count_right_prediction = df[df['instance_state'] == 'RIGHT PREDICTION'].shape[0]
    count_wrong_prediction = df[df['instance_state'] == 'WRONG PREDICTION'].shape[0]
    count_not_detected = df[df['instance_state'] == 'NOT DETECTED'].shape[0]
    # calculate metrics
    precision = count_right_prediction / (count_right_prediction + count_wrong_prediction)
    recall = count_right_prediction / (count_right_prediction + count_not_detected)
    # print metrics
    print("Precision:", precision)
    print("Recall:", recall)
    return precision, recall

def create_pictures(df):
    files = set()
    files = set(df['filename'])
    total_iterations = len(files)
    with tqdm(total=total_iterations, desc="Draw predictions") as pbar:
        for filename in files:
            path_output = 'C:/Users/Paolo/Desktop/Prova2/Traffic Sign Detenction/Traffic Sign Detenction/test/predizioni'
            output_file = path_output + '/right predictions/' + filename
            predictions = pd.DataFrame(columns=columns)
            predictions = df[df['filename'] == filename]
            image = Image.open(predictions.iloc[0]['image_path'])
            draw = ImageDraw.Draw(image)
            correct = True
            for index, prediction in predictions.iterrows():
                if((prediction['instance_state'] == 'RIGHT PREDICTION') or (prediction['instance_state'] == 'WRONG PREDICTION')):
                    if((prediction['instance_state'] == 'WRONG PREDICTION')):
                        output_file = path_output + '/wrong predictions/' + filename
                        correct = False
                    color = random_color()
                    draw.rectangle(prediction['cords_predict'], outline=color, width=5)
                    c1, c2, c3, c4 = prediction['cords_predict']
                    position_label = (c3 + 5, c2)
                    position_label2 = (c3 + 5, c2 + 30)
                    font = ImageFont.truetype("arial.ttf", 20)
                    label = 'Name: ' + prediction['class_name_predict'] + '\nConf: ' + str(round(prediction['conf_classify_predict'], 4))
                    draw.text(position_label, label, font=font, fill=color, stroke_width=1)
                    #print(prediction)
                    #print(type(prediction['image_path']))
                    #draw_bound(prediction['image_path'], prediction['filename'], prediction['cords_predict'], "red", path_output)
            image.save(output_file)
            pbar.update(1)

def random_color():
    colori_disponibili = ["red", "green", "yellow", "blue", "purple", "brown"]
    colore_casuale = random.choice(colori_disponibili)
    return colore_casuale


