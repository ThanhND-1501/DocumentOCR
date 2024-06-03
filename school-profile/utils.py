import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

from ultralytics import YOLO
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def convert_pdf_to_jpg(pdf_path, img_folder):
    os.makedirs(img_folder, exist_ok=True)
    img = convert_from_path(pdf_path, 350, poppler_path=os.path.join('..' ,'poppler-24.02.0', 'Library', 'bin'))
    if len(img) == 1:
        img[0].save(f'{os.path.join(img_folder, os.path.basename(pdf_path)[:-4], os.path.basename(pdf_path)[:-4])}.jpg', 'JPEG')
    else:
        for i in range(len(img)):
            page_folder = os.path.join(img_folder, os.path.basename(pdf_path)[:-4] + '(' + str(i) + ')')
            os.makedirs(page_folder, exist_ok=True)
            img[i].save(f"{os.path.join(page_folder, 'img.jpg')}", 'JPEG')

class ConfigLoader:
    @staticmethod
    def load_classes(classes_file):
        with open(classes_file) as f:
            return [c.strip() for c in f.readlines()]

    @staticmethod
    def load_params(params_file):
        params = {}
        with open(params_file) as f:
            for line in f.readlines():
                line = line.strip().split('=')
                params[line[0]] = eval(line[1])
        return params

class ImageProcessor:
    def __init__(self, model_name, classes, params, save_dir, device="cpu"):
        self.model = YOLO(model_name, task='detect')
        self.classes = classes
        self.params = params
        self.save_dir = save_dir
        self.device = device

        # Load the extractor model only once
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = self.device
        self.extractor = Predictor(config)

    def process_image(self, img_path):
        results_folder = os.path.dirname(img_path) # results/test/test(0)
        
        image = cv2.imread(img_path)
        results = self.model(img_path)

        boxes = results[0].boxes.xyxy.tolist()
        for i in range(len(boxes)):
            boxes[i].insert(0, int(results[0].boxes.cls[i]))
            boxes[i].append(float(results[0].boxes.conf[i]))

        dict_boxes = {}
        for box in boxes:
            if dict_boxes.get(box[0], [-1])[-1] < box[-1]:
                dict_boxes[box[0]] = box[1:]

        norm_boxes = [[key] + dict_boxes[key] for key in dict_boxes.keys()]

        table_box = self._extract_table_box(norm_boxes)
        if table_box:
            self._process_table(image, table_box, img_path, results_folder)
        
        self._extract_image_content(image, norm_boxes, img_path, results_folder)

    def _extract_table_box(self, norm_boxes):
        for i in range(len(norm_boxes)):
            if norm_boxes[i][0] in range(26, 29):
                return norm_boxes.pop(i)
        return None

    def _process_table(self, image, table_box, img_path, results_folder):
        x1, y1, x2, y2 = map(round, table_box[1:5])
        table_img = image[y1:y2, x1:x2]

        table_type = self.classes[table_box[0]]
        parameters = self.params[table_type]

        detector = TableDetector(parameters)
        df_table, transformed_table = detector.detect_table_in_image(table_img, results_folder, os.path.basename(img_path)[:-4] + '_table')
        lb_arr = df_table.to_numpy()

        content_table = self._extract_table_content(lb_arr, transformed_table)
        df_content_table = pd.DataFrame(content_table)
        df_content_table.to_csv(os.path.join(results_folder, os.path.basename(img_path)[:-4] + '_content_table.csv'), index=False, header=False)

    def _extract_table_content(self, lb_arr, transformed_table):
        content_table = []
        for row in range(len(lb_arr)):
            row_content = []
            for col in range(len(lb_arr[row])):
                if lb_arr[row][col] in [np.nan, None]:
                    row_content.append(np.nan)
                    continue
                x, y, w, h = lb_arr[row][col][:4]
                subimg = transformed_table[y:y + h, x:x + w]
                content = self.extractor.predict(Image.fromarray(subimg))
                row_content.append(content)
            content_table.append(row_content)
        return content_table

    def _extract_image_content(self, image, norm_boxes, img_path, results_folder):
        norm_boxes.sort()

        content_image = {}
        for box in norm_boxes:
            field = self.classes[box[0]]
            x1, y1, x2, y2 = map(round, box[1:5])
            tmp_img = image[y1:y2, x1:x2]
            content = self.extractor.predict(Image.fromarray(tmp_img))
            content_image[field] = content
        
        df_content = pd.DataFrame(content_image, index=[0])
        df_content.to_csv(os.path.join(results_folder, os.path.basename(img_path)[:-4] + '_content.csv'))

class TableDetector:
    def __init__(self, params):
        self.params = params

    def preprocess_image(self, img):
        if isinstance(img, str):
            original = cv2.imread(img)
        else:
            original = img
        gray_scale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        strip = self.params[4]
        if strip != 0:
            strip_val = np.mean(gray_scale)
            gray_scale[:strip, :], gray_scale[-strip:, :], gray_scale[:, :strip], gray_scale[:, -strip:] = strip_val, strip_val, strip_val, strip_val

        blur = cv2.GaussianBlur(gray_scale, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 3)
        return original, gray_scale, thresh

    def find_contours(self, thresh):
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        return max(cnts, key=cv2.contourArea)

    def get_transformed_image(self, gray_scale, image, rect):
        (tl, tr, br, bl) = rect
        x, y, w, h = cv2.boundingRect(rect)
        dst = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_gray = cv2.warpPerspective(gray_scale, M, (image.shape[1], image.shape[0]))
        transformed_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
        return transformed_image, warped_gray, w, h

    def otsu_canny(self, image, lowrate=0.1):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edged = cv2.Canny(image, int(ret * lowrate), ret)
        return edged

    def detect_boxes(self, img_bin, table_width, table_height):
        dil_kernel = np.ones((3, 3), np.uint8)
        img_bin = cv2.dilate(img_bin, dil_kernel, iterations=2)

        line_ratio_width, line_ratio_height, box_ratio_width, box_ratio_height = self.params[:4]

        line_min_width = int(img_bin.shape[1] * line_ratio_width)
        line_min_height = int(img_bin.shape[0] * line_ratio_height)

        kernal_h = np.ones((1, line_min_width), np.uint8)
        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h, iterations=1)

        kernal_v = np.ones((line_min_height, 1), np.uint8)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v, iterations=1)

        img_bin_final = img_bin_h | img_bin_v

        final_kernel = np.ones((3, 3), np.uint8)
        img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        box_list = [
            (x, y, w, h, area)
            for x, y, w, h, area in stats[2:]
            if area > 100 and h > box_ratio_height * table_height and w > box_ratio_width * table_width and w * h < 0.3 * img_bin.shape[0] * img_bin.shape[1]
        ]
        
        return np.array(box_list, dtype=[('x', int), ('y', int), ('w', int), ('h', int), ('area', int)])

    def sort_and_norm_boxes(self, box_list):
        order = ['y', 'x']
        for j in range(2):
            tmp = 1e10
            start = 0
            end = 1
            box_list.sort(order=order[j])

            for i in range(len(box_list)):
                diff = abs(box_list[i][order[j]] - tmp)
                if diff > 10:
                    box_list[start:end][order[j]] = [round(np.mean(box_list[start:end][order[j]]))] * (end - start)
                    start = i
                end = i + 1
                tmp = box_list[i][order[j]]
            try:
                box_list[start:end][order[j]] = [round(np.mean(box_list[start:end][order[j]]))] * (end - start)
            except:
                continue

        box_list.sort(order=order[::-1])
        return box_list
    
    def arrange_boxes(self, box_list):
        cols = {}
        row = set()

        for box in box_list:
            col = box[0]
            row.add(int(box[1]))
            cols[col] = cols.get(col, [])
            cols[col].append([int(x) for x in box])
        keys = list(cols.keys())
        row = sorted(list(row))

        for i in range(len(row)):
            for key in keys:
                try:
                    if cols[key][i][1] > row[i]:
                        cols[key].insert(i, None)
                except IndexError:
                    cols[key].append(None)
        df = pd.DataFrame.from_dict(cols, orient='index').transpose()
        return df

    def save_table(self, transformed_image, df, output_path):
        cv2.imwrite(output_path + '.jpg', transformed_image)
        df.to_csv(output_path + '.csv', index=False, header=False)
        print('Saved table in', output_path)

    def detect_table_in_image(self, img, output_folder, output_table_path):
        original, gray_scale, thresh = self.preprocess_image(img)
        c = self.find_contours(thresh)
        pts = c.squeeze(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        transformed_image, warped_gray, table_width, table_height = self.get_transformed_image(gray_scale, original, rect)
        img_bin1 = self.otsu_canny(warped_gray, self.params[5])
        box_list = self.detect_boxes(img_bin1, table_width, table_height)
        sorted_box_list = self.sort_and_norm_boxes(box_list)

        clone = transformed_image.copy()
        for box in sorted_box_list:
            cv2.rectangle(clone, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder, 'table_cells.jpg'), clone)

        df_table = self.arrange_boxes(sorted_box_list)
        print('Output path:', output_table_path)
        self.save_table(transformed_image, df_table, os.path.join(output_folder, output_table_path))

        return df_table, transformed_image
