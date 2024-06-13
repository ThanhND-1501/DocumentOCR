import argparse
import os
from utils import ImageTextDetector, convert_pdf_to_jpg

def main(input_path, savedir, device):
    if input_path.endswith('.jpg'):
        os.makedirs(os.path.join(savedir, os.path.basename(input_path)[:-4]))
    elif input_path.endswith('.pdf'):
        convert_pdf_to_jpg(input_path, savedir)
        input_path = os.path.join(savedir, os.path.basename(input_path)[:-4])

    detector = ImageTextDetector(input_path, savedir, device)
    detector.run_inference()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text detection on images.")
    parser.add_argument("--input_path", type=str, help="Path to the input image or folder of images.")
    parser.add_argument("--savedir", type=str, default='results', help="Path to the directory where results will be saved.")
    parser.add_argument('--device', type=str, help='Device to use.', default='cpu')
    args = parser.parse_args()

    main(args.input_path, args.savedir, args.device)
