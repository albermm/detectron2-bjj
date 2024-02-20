import cv2
import torch
import numpy as np
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

        if model_type == "OD":  # Object detection
            self.cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            )

        elif model_type == "IS":  # Instance segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )

        elif model_type == "KP":  # Keypoint Detection
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
                )
            )
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
            )

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, frame):
        # Use the predictor to get instance predictions
        with torch.no_grad():
            outputs = self.predictor(frame)["instances"]

        # visualizer without the instance_mode=ColorMode.IMAGE_BW
        v = Visualizer(
            frame[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        output = v.draw_instance_predictions(outputs.to("cpu"))

        # Update frame with the visualized result
        out_frame_seg = output.get_image()[:, :, ::-1]

        return out_frame_seg, outputs

    def onImage(self, input_path, output_path):
        image_path = input_path
        output_path = output_path

        image = cv2.imread(image_path)
        out_frame = image.copy()
        out_frame_seg, outputs = self.predict(image)

        print("Instances Tensor:")
        print(outputs)

        cv2.imwrite(output_path, out_frame_seg)
        return out_frame, out_frame_seg # Placeholder for consistency

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
        print(f"No of frames {n_frames}")
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

            out_frame_seg, outputs = self.predict(frame)

            # Write the frame to the output video
            out.write(out_frame_seg)

            done += 1
            percent = int((done / n_frames) * 100)
            sys.stdout.write(
                "\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent)
            )
            sys.stdout.flush()

        # Release video capture and writer
        cap.release()
        out.release()


        return output_path
