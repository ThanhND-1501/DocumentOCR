import argparse
from utils import TableDetector, ConfigLoader, ImageProcessor

def main(model_name, img_path, classes_file, params_file, save_dir):
    classes = ConfigLoader.load_classes(classes_file)
    params = ConfigLoader.load_params(params_file)

    processor = ImageProcessor(model_name, classes, params, save_dir)
    processor.process_image(img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and extract information from an image.")
    parser.add_argument('--img_path', type=str, help='Path to the image file.')
    parser.add_argument('--model_name', type=str, help='Path to the model file.', default='models\yolov8_best_detection_school_profile.pt')
    parser.add_argument('--classes_file', type=str, help='Path to the classes file.', default='classes.txt')
    parser.add_argument('--params_file', type=str, help='Path to the parameters file.', default='params.txt')
    parser.add_argument('--save_dir', type=str, help='Directory to save the results.', default='results')

    args = parser.parse_args()
    main(args.model_name, args.img_path, args.classes_file, args.params_file, args.save_dir)
