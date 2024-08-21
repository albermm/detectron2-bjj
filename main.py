import json
from argparse import ArgumentParser
from utils.helper import Predictor
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

    args = parser.parse_args()

    # Initialize predictor and load trained model
    predictor = Predictor()
    trained_model = './trained_model.joblib'  # Load your trained model here
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
        print("Call predictor onImage from helper")
        keypoint_frame, densepose_frame, keypoints, densepose = predictor.onImage(input_path, output_path)
        # Call find_position function and store the result
        #predicted_position = find_position(all_pred_keypoints)
        #print(f"Predicted Position: {predicted_position}")
        print("returning from helper")
        print("=======================")
        #print(f'Keypoints: {keypoints}', f'Densepose: {densepose}')



if __name__ == "__main__":
    main()
                