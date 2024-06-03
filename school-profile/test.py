import argparse
import os
from utils import ConfigLoader, ImageProcessor, convert_pdf_to_jpg

classes = ConfigLoader.load_classes('school-profile\classes.txt')
params = ConfigLoader.load_params('school-profile\params.txt')

processor = ImageProcessor('school-profile\models\yolov8_best_detection_school_profile.pt', classes, params, 'results', 'cpu')

img_path = 'school-profile\\test.pdf'
save_dir = 'school-profile\\results'

if img_path.endswith('.jpg'):
    processor.process_image(img_path)
elif img_path.endswith('.pdf'):
    img_folder = os.path.join(save_dir, os.path.basename(img_path)[:-4])
    convert_pdf_to_jpg(img_path, img_folder)
    for img in os.listdir(img_folder):
        img = os.path.join(img_folder, img, 'img.jpg')
        processor.process_image(img)