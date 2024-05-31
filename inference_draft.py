from ultralytics import YOLO
from utils import TableDetector
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

with open('classes.txt') as f:
    txt = f.readlines()
classes = [c.strip() for c in txt]

with open('params.txt') as f:
    params = {}
    for line in f.readlines():
        line = line.strip().split('=')
        params[line[0]] = eval(line[1])

model_name = 'models\yolov8_best_detection_school_profile.pt'
img_path = 'test.jpg'
results_folder = os.path.join('results', os.path.basename(img_path[:-4]))
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

for i in range(len(norm_boxes)):
    if norm_boxes[i][0] in range(26, 29):
        table_box = norm_boxes.pop(i)
        break

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

# Extract table
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

# Extract content
content_image = {}
norm_boxes.sort()
for box in norm_boxes:
    field = classes[box[0]]
    x1, y1, x2, y2 = map(round, box[1:5])
    tmp_img = image[y1:y2,x1:x2]
    content = extractor.predict(Image.fromarray(tmp_img))
    content_image[field] = content
df_content = pd.DataFrame(content_image, index=[0])
df_content.to_csv(os.path.join(results_folder, os.path.basename(img_path)[:-4]+'_content.csv'))