"""
A method to predict instances in a frame and visualize the predictions.

Parameters:
    frame: The input frame for prediction

Returns:
    out_frame: The updated frame with visualized results
    outputs: The instance predictions
"""

import cv2
import torch
import sys
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from utils.find_position import find_position, find_positions_video

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.alpha = 0.5
        self.last_outputs = None
        self.out_video = None 

        # load model config and pretrained model
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        # Define keyframe threshold percentage
        self.keyframe_threshold_percentage = 0.6

    def predict(self, frame):
        # Use the predictor to get instance predictions
        with torch.no_grad():
            outputs = self.predictor(frame)["instances"]

        # Visualize predictions on the frame
        v = Visualizer(
            frame[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        output = v.draw_instance_predictions(outputs.to("cpu"))


        # Update frame with the visualized result
        out_frame = output.get_image()[:, :, ::-1]

        return out_frame, outputs


    def process_frame_with_smoothing(self, frame, alpha=0.5):
        # Convenience method to process a single frame with temporal smoothing
        out_frame, outputs = self.predict(frame)
        self.last_outputs = outputs
        
        if hasattr(self, 'last_outputs') and outputs is not None:
            # Calculate mean confidence and set keyframe threshold
            confidence_scores = outputs.scores
            mean_confidence = sum(confidence_scores) / len(confidence_scores)
            keyframe_threshold = self.keyframe_threshold_percentage * mean_confidence

            # Apply temporal smoothing using a simple moving average
            smoothed_keypoints = (
                alpha * np.array(outputs.pred_keypoints) + (1 - alpha) * np.array(self.last_outputs.pred_keypoints)
            )

            # Update last_outputs for the next frame
            self.last_outputs = outputs

            # Check if the frame is a keyframe based on the calculated threshold
            if self.is_keyframe(outputs, keyframe_threshold):
                print("Keyframe detected!")

            return outputs
        else:
            # Handle the case when outputs are None or last_outputs is not initialized
            print("Outputs are None or last_outputs not initialized for frame.")
            return None

    def is_keyframe(self, outputs, threshold=None):
        if threshold is None:
            threshold = self.keyframe_threshold_percentage

        # Check if any instance has confidence above the threshold
        for score in outputs.scores:
            if score > threshold:
                return True

        return False


    def save_outputs(self, outputs):
        instances = outputs

        # Check if 'pred_keypoints' attribute exists in 'instances'
        if hasattr(instances, 'pred_keypoints'):
            # Extract pred_keypoints
            pred_keypoints = instances.pred_keypoints

            # List to store pred_keypoints for all instances
            all_pred_keypoints = []

            # Iterate through each instance
            for i in range(len(pred_keypoints)):
                # Access pred_keypoints for the current instance
                instance_pred_keypoints = pred_keypoints[i]

                # Convert pred_keypoints to a Python list
                pred_keypoints_list = instance_pred_keypoints.tolist()

                # Append to the list of all pred_keypoints
                all_pred_keypoints.append(pred_keypoints_list)

            # Save the result to a JSON file
            #json_filename = "all_pred_keypoints.json"

            #with open(json_filename, 'w') as json_file:
             #   json.dump(all_pred_keypoints, json_file)

            #print(f"All pred keypoints saved to {json_filename}")
            #print(f' los keypoints son: {all_pred_keypoints}')
            return all_pred_keypoints
        
        else:
            print("The 'pred_keypoints' attribute is not present in the given Instances object.")

        


    def onImage(self, input_path, output_path):
        image_path = input_path
        output_path = output_path

        image = cv2.imread(image_path)
        out_frame, outputs = self.predict(image)

        cv2.imwrite(output_path, out_frame)
        print(f"Processed image saved at: {output_path}")

        # Save the outputs to a JSON file
        all_pred_keypoints = self.save_outputs(outputs)
        
        return out_frame, outputs, all_pred_keypoints

    def process_video(self, video_path, output_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error opening the video file...")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f'num frames {n_frames}')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Prepare VideoWriter for output video
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Create a VideoVisualizer
            visualizer = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

            # List to store detected positions
            positions_list = []

            done = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process each frame
                out_frame, outputs = self.predict(frame)

                # Save the frame with predictions
                out_video.write(out_frame)
                
                # Extract keypoints for the current frame
                frame_keypoints = self.save_outputs(outputs)

                # If the save_outputs function returns keypoints, proceed to find the position
                if frame_keypoints:
                    predicted_position = find_position(frame_keypoints)
                    
                    positions_list.append({"frame_number": done, "keypoints": frame_keypoints, "position": predicted_position})
                    print(f"Frame {done} - Position: {predicted_position}")
                else:
                    # Handle the case when keypoints are not available for the current frame
                    print(f"Keypoints not available for frame {done}")
                # Find position for the current frame using your existing logic
                #predicted_position = find_position(frame_keypoints)

                # Append the detected position to the list
                #positions_list.append({"frame_number": done, "keypoints": frame_keypoints.tolist(), "position": predicted_position.tolist()})

                done += 1

            # Release video capture and writer
            cap.release()
            out_video.release()
            return positions_list

            # Enrich the annotations JSON file with new data
            #self.enrich_annotations(annotations_path, positions_list)

    def process_video_with_keyframes(self, video_path, output_path, keyframe_threshold=0.1):
          cap = cv2.VideoCapture(video_path)
          if not cap.isOpened():
              print("Error opening the video file...")
              return None

          fps = cap.get(cv2.CAP_PROP_FPS)
          n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          print(f'num frames {n_frames}')
          width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

          # Prepare VideoWriter for output video
          fourcc = cv2.VideoWriter_fourcc(*"XVID")
          out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

          # Create a VideoVisualizer
          visualizer = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

          # List to store detected positions
          positions_list_modified = []
          done = 0
          while True:
              ret, frame = cap.read()
              if not ret:
                  break

              # Process each frame with temporal smoothing
              outputs = self.process_frame_with_smoothing(frame)

              # Extract keypoints for the current frame
              frame_keypoints = self.save_outputs(outputs)

              # If the save_outputs function returns keypoints, proceed to find the position
              if frame_keypoints:
                  predicted_position = find_position(frame_keypoints)

                  # Check if the frame is a keyframe based on some threshold
                  if self.is_keyframe(outputs, keyframe_threshold):
                      positions_list_modified.append({"frame_number": done, "keypoints": frame_keypoints, "position": predicted_position})
                      print(f"Frame {done} - Position: {predicted_position} (Keyframe)")
                  else:
                      print(f"Frame {done} - Position: {positions_list_modified}")
              else:
                  print(f"Keypoints not available for frame {done}")

              done += 1

          # Release video capture and writer
          cap.release()
          out_video.release()
          return positions_list_modified

