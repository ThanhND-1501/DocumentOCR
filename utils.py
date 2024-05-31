import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TableDetector:
    def __init__(self, params):
        self.params = params

    def preprocess_image(self, img):
        if img is str:
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
        dst = np.array([(x, y), (x+w, y), (x+w, y+h), (x, y+h)], dtype="float32")
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
            if area > 100 and h > box_ratio_height * table_height and w > box_ratio_width * table_width and w*h < 0.3*img_bin.shape[0]*img_bin.shape[1]
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
        cv2.imwrite(output_path+'.jpg', transformed_image)
        df.to_csv(output_path+'.csv', index=False, header=False)
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
            cv2.rectangle(clone, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder, 'table_cells.jpg'), clone)

        df_table = self.arrange_boxes(sorted_box_list)
        self.save_table(transformed_image, df_table, os.path.join(output_folder, output_table_path))

        return df_table, transformed_image
