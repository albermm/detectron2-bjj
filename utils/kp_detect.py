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
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()

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

    def process_frame(self, frame):
        # Convenience method to process a single frame
        return self.predict(frame)

    def onImage(self, input_path, output_path):
        image_path = input_path
        output_path = output_path

        image = cv2.imread(image_path)
        out_frame, out_frame = self.process_frame(image)

        cv2.imwrite(output_path, out_frame)
        return out_frame, out_frame

    def onVideo(self, input_path, output_path):
        video_path = input_path
        output_path = output_path

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error opening the video file...")
            return None

        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter for output
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process each frame
        done = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_frame, out_frame = self.process_frame(frame)

            # Write the frame to the output video
            out.write(out_frame)

            done += 1
            percent = int((done / n_frames) * 100)
            sys.stdout.write("\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent))
            sys.stdout.flush()

        # Release video capture and writer
        cap.release()
        out.release()

        return output_path



