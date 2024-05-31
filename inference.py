import argparse
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils import TableDetector
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def load_classes(classes_file):
    with open(classes_file) as f:
        return [c.strip() for c in f.readlines()]

def load_params(params_file):
    params = {}
    with open(params_file) as f:
        for line in f.readlines():
            line = line.strip().split('=')
            params[line[0]] = eval(line[1])
    return params

def main(model_name, img_path, classes_file, params_file, save_dir):
    classes = load_classes(classes_file)
    params = load_params(params_file)

    results_folder = os.path.join(save_dir, os.path.basename(img_path[:-4]))
    os.makedirs(results_folder, exist_ok=True)

    model = YOLO(model_name, task='detect')
    image = cv2.imread(img_path)
    results = model(img_path)

    boxes = results[0].boxes.xyxy.tolist()
    for i in range(len(boxes)):
        boxes[i].insert(0, int(results[0].boxes.cls[i]))
        boxes[i].append(float(results[0].boxes.conf[i]))

    dict_boxes = {}
    for box in boxes:
        if dict_boxes.get(box[0], [-1])[-1] < box[-1]:
            dict_boxes[box[0]] = box[1:]

    norm_boxes = [[key] + dict_boxes[key] for key in dict_boxes.keys()]

    table_box = None
    for i in range(len(norm_boxes)):
        if norm_boxes[i][0] in range(26, 29):
            table_box = norm_boxes.pop(i)
            break

    if table_box:
        x1, y1, x2, y2 = map(round, table_box[1:5])
        table_img = image[y1:y2, x1:x2]

        table_type = classes[table_box[0]]
        parameters = params[table_type]

        detector = TableDetector(parameters)
        df_table, transformed_table = detector.detect_table_in_image(table_img, results_folder, img_path[:-4]+'_table')
        lb_arr = df_table.to_numpy()

        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        extractor = Predictor(config)

        # Extract table content
        content_table = []
        for row in range(len(lb_arr)):
            row_content = []
            for col in range(len(lb_arr[row])):
                if lb_arr[row][col] in [np.nan, None]:
                    row_content.append(np.nan)
                    continue
                x, y, w, h = lb_arr[row][col][:4]
                subimg = transformed_table[y:y+h, x:x+w]
                content = extractor.predict(Image.fromarray(subimg))
                row_content.append(content)
            content_table.append(row_content)
        df_content_table = pd.DataFrame(content_table)
        df_content_table.to_csv(os.path.join(results_folder, os.path.basename(img_path)[:-4]+'_content_table.csv'), index=False, header=False)

    # Extract image content
    content_image = {}
    norm_boxes.sort()
    for box in norm_boxes:
        field = classes[box[0]]
        x1, y1, x2, y2 = map(round, box[1:5])
        tmp_img = image[y1:y2, x1:x2]
        content = extractor.predict(Image.fromarray(tmp_img))
        content_image[field] = content
    df_content = pd.DataFrame(content_image, index=[0])
    df_content.to_csv(os.path.join(results_folder, os.path.basename(img_path)[:-4]+'_content.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract information from an image.")
    parser.add_argument('--img_path', type=str, help='Path to the image file.')
    parser.add_argument('--model_name', type=str, help='Path to the model file.', default='models\yolov8_best_detection_school_profile.pt')
    parser.add_argument('--classes_file', type=str, help='Path to the classes file.', default='classes.txt')
    parser.add_argument('--params_file', type=str, help='Path to the parameters file.', default='params.txt')
    parser.add_argument('--save_dir', type=str, help='Directory to save the results.', default='results')

    args = parser.parse_args()
    main(args.model_name, args.img_path, args.classes_file, args.params_file, args.save_dir)
