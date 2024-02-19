import cv2
from utils.helper import Detector
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--input", type=str, help="Set the input path to the video", required=True
)
parser.add_argument(
    "--out", type=str, help="Set the output path to the video", required=True
)
parser.add_argument(
    "--model_type", type=str, help="Set the model type (OD, IS, KP)", required=True
)
args = parser.parse_args()

# Create the predictor with the specified model_type
predictor = Detector(model_type=args.model_type)

if __name__ == "__main__":
    # Example usage for processing a single image
    image_path = args.input
    out_frame, out_frame_seg = predictor.onImage(image_path)
    
    # Now you can do something with the results, such as displaying or saving them

    # Write the frame to the output image
    output_image_path = args.out
    cv2.imwrite(output_image_path, out_frame_seg)

