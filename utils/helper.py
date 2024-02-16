import logging
import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

class GetLogger:
    @staticmethod
    def logger(name):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(name)

class Detector:
    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()

        #load model config and pretrained model
        
        if model_type == "OD": #Object detection
          self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
          self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        elif model_type == "IS": #Instance segmentation
          self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
          self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        elif model_type == "KP": #Keypoint Detection
           self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
           self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)

    
    
    def onImage(self, image_path):
        image = cv2.imread(image_path)
        # Use the predictor to get instance predictions
        with torch.no_grad():
            outputs = self.predictor(image)["instances"]

        
        # Create copies of the input frame for visualization
        out_frame = image.copy()
        out_frame_seg = np.zeros(out_frame.shape, dtype=out_frame.dtype)

        # Visualize the instance predictions
        viz = Visualizer(out_frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         instance_mode=ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(outputs.to("cpu"))

        # Update out_frame_seg with the visualized result
        out_frame_seg = output.get_image()[:, :, ::-1]

        #cv2.imshow("result", output.get_image()[:, :, ::-1])
        print("Instances Tensor:")
        print(outputs)

        return out_frame, out_frame_seg