import cv2
import os
import torch
import json
import numpy as np
import yaml
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from utils.find_position import find_position

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class GetLogger:
    @staticmethod
    def logger(name):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(name)

class Predictor:
    def __init__(self):
        self.cfg_kp = get_cfg()
        self.cfg_dp = get_cfg()
        self.alpha = 0.5
        self.last_outputs = None
        self.out_video = None 

        # Get the base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load Keypoint model config and pretrained model
        self.cfg_kp.merge_from_file(model_zoo.get_config_file(config['keypoint_model']['config']))
        self.cfg_kp.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['keypoint_model']['weights'])
        self.cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['keypoint_score_threshold']
        self.cfg_kp.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_kp = DefaultPredictor(self.cfg_kp)
        
        # Load DensePose model config and pretrained model
        add_densepose_config(self.cfg_dp)
        model_configs_path = os.path.join(base_dir, config['densepose_model']['config'])
        models_path = os.path.join(base_dir, config['densepose_model']['weights'])
        
        self.cfg_dp.merge_from_file(model_configs_path)
        self.cfg_dp.MODEL.WEIGHTS = models_path
        self.cfg_dp.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg_dp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['densepose_score_threshold']
        self.predictor_dp = DefaultPredictor(self.cfg_dp)
        self.extractor = DensePoseResultExtractor()
        self.visualizer = DensePoseResultsFineSegmentationVisualizer()

    def predict_keypoints(self, frame):
        print("called predict_kepoints")
        with torch.no_grad():
            outputs = self.predictor_kp(frame)["instances"]

        v = DetectronVisualizer(
            frame[:, :, ::-1],
            MetadataCatalog.get(self.cfg_kp.DATASETS.TRAIN[0]),
            scale=1.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        output = v.draw_instance_predictions(outputs.to("cpu"))

        out_frame = output.get_image()[:, :, ::-1]
        print("outframe is ready and returned - end of predict_keypoints")
        return out_frame, outputs

    def save_keypoints(self, outputs):
        instances = outputs

        if hasattr(instances, 'pred_keypoints'):
            pred_keypoints = instances.pred_keypoints
            all_pred_keypoints = [keypoints.tolist() for keypoints in pred_keypoints]
            print(f'The keypoints are: {all_pred_keypoints}')
            return all_pred_keypoints
        else:
            print("The 'pred_keypoints' attribute is not present in the given Instances object.")
            return None

    def onImage(self, input_path, output_path):
        try:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Failed to load image from {input_path}")

            keypoint_frame, keypoint_outputs = self.predict_keypoints(image)
            cv2.imwrite(output_path + "_keypoints.jpg", keypoint_frame)
     
            keypoints = self.save_keypoints(keypoint_outputs)  
            if keypoints is None:
                raise ValueError("Failed to extract keypoints")

            with open(output_path + "_keypoints.json", 'w') as f:
                json.dump(keypoints, f)
       
            predicted_position = find_position(keypoints)

            return keypoint_frame, keypoints, predicted_position
        except Exception as e:
            print(f"Error in onImage: {str(e)}")
            return None, None, None