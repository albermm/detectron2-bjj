from argparse import ArgumentParser
from utils.kp_detect import Detector
from utils.matching import match_keypoints, assign_players
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
    trained_model = None  # Load your trained model here

    input_path = args.input
    output_path = args.out

    if input_path.lower().startswith('https://www.youtube.com'):
        # Process YouTube video
        # (You can implement this part based on your previous modifications)
        pass
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        predictions = predictor.onVideo(input_path, output_path)

        # Check if video processing was successful
        if predictions:
            print(f"Processed video saved at: {output_path}")

            # Load annotations from the file
            annotations = []  # Load your annotations here

            # Match keypoints and assign players
            matching_result = match_keypoints(trained_model, annotations, predictions)

            # Now 'matching_result' contains a summary of matches for each frame
            # Analyze 'matching_result' to evaluate the model's performance

        else:
            print("Video processing failed.")
    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Process image
        out_frame, out_frame_seg = predictor.onImage(input_path, output_path)

        # Save the segmented frame to the output path
        cv2.imwrite(output_path, out_frame_seg)
        print(f"Processed image saved at: {output_path}")
    else:
        print("Unsupported input format. Please provide a video (.mp4, .avi, .mov), an image (.jpg, .jpeg, .png), or a YouTube link.")

if __name__ == "__main__":
    main()
