import argparse
import os
from utils import ConfigLoader, ImageProcessor, convert_pdf_to_jpg

def main(model_name, img_path, classes_file, params_file, save_dir, device):
    classes = ConfigLoader.load_classes(classes_file)
    params = ConfigLoader.load_params(params_file)

    processor = ImageProcessor(model_name, classes, params, save_dir, device)

    if img_path.endswith('.jpg'):
        processor.process_image(img_path)
    elif img_path.endswith('.pdf'):
        img_folder = os.path.join(save_dir, os.path.basename(img_path)[:-4])
        convert_pdf_to_jpg(img_path, img_folder)
        for img in os.listdir(img_folder):
            img = os.path.join(img_folder, img, 'img.jpg')
            processor.process_image(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract information from an image.")
    parser.add_argument('--img_path', type=str, help='Path to the image file.')
    parser.add_argument('--model_name', type=str, help='Path to the model file.', default='models\yolov8_best_detection_school_profile.pt')
    parser.add_argument('--classes_file', type=str, help='Path to the classes file.', default='classes.txt')
    parser.add_argument('--params_file', type=str, help='Path to the parameters file.', default='params.txt')
    parser.add_argument('--save_dir', type=str, help='Directory to save the results.', default='results')
    parser.add_argument('--device', type=str, help='Device to use.', default='cpu')

    args = parser.parse_args()
    main(args.model_name, args.img_path, args.classes_file, args.params_file, args.save_dir, args.device)
