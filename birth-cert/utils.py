import os
import cv2
import imutils
import pandas as pd
from PIL import Image
from imutils.contours import sort_contours
from pdf2image import convert_from_path

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def convert_pdf_to_jpg(pdf_path, img_folder):
    os.makedirs(img_folder, exist_ok=True)
    img = convert_from_path(pdf_path, 200, poppler_path=os.path.join('..' ,'poppler-24.02.0', 'Library', 'bin'))
    if len(img) == 1:
        img_folder = os.path.join(img_folder, os.path.basename(pdf_path)[:-4])
        os.makedirs(img_folder, exist_ok=True)
        img[0].save(f'{os.path.join(img_folder, os.path.basename(pdf_path)[:-4])}.jpg', 'JPEG')
    else:
        for i in range(len(img)):
            page_folder = os.path.join(img_folder, os.path.basename(pdf_path)[:-4] + '(' + str(i) + ')')
            os.makedirs(page_folder, exist_ok=True)
            img[i].save(f"{os.path.join(page_folder, 'img.jpg')}", 'JPEG')

class ImageTextDetector:
    def __init__(self, input_path, savedir, device, config_name='vgg_transformer'):
        self.input_path = input_path
        self.savedir = savedir
        self.device = device
        self.predictor = self.load_predictor(config_name)
    
    def load_predictor(self, config_name):
        config = Cfg.load_config_from_name(config_name)
        config['device'] = self.device
        predictor = Predictor(config)
        return predictor

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        (H, W, _) = image.shape
        resized = cv2.resize(image, (int(2*W), int(2*H)), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]
        
        blackhatKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhatKernel)
        
        dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 10))
        grad = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, dilate)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, (3, 2), iterations=1)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        boxes = sort_contours(cnts, method="top-to-bottom")[0]
        
        clone = resized.copy()
        boxes_filtered = []
        for c in boxes:
            (x, y, w, h) = cv2.boundingRect(c)
            if w*h > 600 and w > 3*h and h > 20:
                boxes_filtered.append((x, y, w, h))
        
        # Sort boxes by rows and then by columns within each row
        boxes_filtered = sorted(boxes_filtered, key=lambda b: (b[1] // 10, b[0]))
        
        cropped = []
        pad = 10
        for (x, y, w, h) in boxes_filtered:
            cropped.append(Image.fromarray(cv2.cvtColor(resized[max(0, y-pad):(y+h+1)+pad, max(0, x-pad):(x+w+1)+pad, :],
                                                        cv2.COLOR_BGR2RGB)))
            clone = cv2.rectangle(clone, (max(0, x-pad), max(0, y-pad)), ((x+w+1)+pad, (y+h+1)+pad), (0, 255, 0), 6)
        
        self.save_image(image_path, clone)
        text = self.extract_text(cropped)
        return text
    
    def save_image(self, image_path, clone):
        os.makedirs(self.savedir, exist_ok=True)
        image_name = os.path.join(self.savedir, os.path.basename(image_path)[:-4], os.path.basename(image_path)[:-4] + '_detected.jpg')
        cv2.imwrite(image_name, clone)
    
    def extract_text(self, cropped_images):
        text = []
        for img in cropped_images:
            text.append(self.predictor.predict(img))
        return text
    
    def process_and_save(self, image_path):
        text = self.process_image(image_path)
        image_name = os.path.basename(image_path)
        csv_path = os.path.join(self.savedir, os.path.splitext(image_name)[0], f'{os.path.splitext(image_name)[0]}.csv')
        df = pd.DataFrame([text])
        df.to_csv(csv_path, index=False)
    
    def run_inference(self):
        if os.path.isdir(self.input_path):
            files = [f for f in os.listdir(self.input_path) if f.lower().endswith('.jpg')]
            for item in files:
                img_path = os.path.join(self.input_path, item)
                self.process_and_save(img_path)
        elif os.path.isfile(self.input_path) and self.input_path.lower().endswith('.jpg'):
            self.process_and_save(self.input_path)
        else:
            print("Invalid input path. Please provide a valid image file or a folder containing image files.")
