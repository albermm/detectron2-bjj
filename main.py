import json
from argparse import ArgumentParser
from utils.kp_detect import Detector
from utils.find_position import find_position
#from utils.matching import match_keypoints
#from utils.assign_players import assign_players
#from utils.model_trainer import load_annotations
import cv2
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="Set the input path to the video, image, or YouTube link", required=True
    )
    parser.add_argument(
        "--out", type=str, help="Set the output path to the video or image", required=True
    )
    parser.add_argument(
        "--model_type", type=str, help="Set the model type (OD, IS, KP)", required=True
    )
    args = parser.parse_args()

    # Initialize predictor and load trained model
    predictor = Detector(model_type=args.model_type)
    trained_model = '/content/detectron2-bjj/trained_model.joblib'  # Load your trained model here
    #json_file_path = '/content/annotations.json'
    #annotations = load_annotations(json_file_path)    
    input_path = args.input
    output_path = args.out

    if input_path.lower().startswith('https://www.youtube.com'):
        # Process YouTube video
        # (You can implement this part based on your previous modifications)
        pass

    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        postions_list = predictor.process_video_with_keyframes(input_path, output_path)

    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Process image to detect Key points
        out_frame, outputs, all_pred_keypoints = predictor.onImage(input_path, output_path)
        # Call find_position function and store the result
        predicted_position = find_position(all_pred_keypoints)
        print(f"Predicted Position: {predicted_position}")



if __name__ == "__main__":
    main()
                