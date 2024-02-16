import cv2
from utils.helper import Detector
from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--input", type=str, help="Set the input path to the video", required=True
)
parser.add_argument(
    "--out", type=str, help="Set the output path to the video", required=True
)
args = parser.parse_args()



predictor = Detector(model_type="KP")



if __name__ == "__main__":
    # Example usage for processing a single image
    predictor = Detector()
    image_path = args.input
    out_frame, out_frame_seg = predictor.onImage(image_path)
    # Now you can do something with the results, such as displaying or saving them


    # Write the frame to the output image
    output_image_path = args.out
    cv2.imwrite(output_image_path, out_frame_seg)


