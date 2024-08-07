import cv2
import os
import torch
import json
import numpy as np
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.utils.video_visualizer import VideoVisualizer

from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer

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

        # Load Keypoint model config and pretrained model
        self.cfg_kp.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg_kp.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg_kp.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_kp = DefaultPredictor(self.cfg_kp)
        
        # Load DensePose model config and pretrained model
        add_densepose_config(self.cfg_dp)
        self.cfg_dp.merge_from_file("model_configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        self.cfg_dp.MODEL.WEIGHTS = "models/model_final_162be9.pkl"
        #Update paths to be relative to the current script location
        #script_dir = os.path.dirname(__file__)
        #model_configs_path = os.path.join(script_dir, 'model_configs', 'densepose_rcnn_R_50_FPN_s1x.yaml')
        #models_path = os.path.join(script_dir, 'models', 'model_final_162be9.pkl')

        self.cfg_dp.merge_from_file(model_configs_path)
        self.cfg_dp.MODEL.WEIGHTS = models_path
        self.cfg_dp.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg_dp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor_dp = DefaultPredictor(self.cfg_dp)
        self.extractor = DensePoseResultExtractor()
        self.visualizer = DensePoseResultsFineSegmentationVisualizer()

    def predict_keypoints(self, frame):
        
        print("called predict_kepoints")
        # Use the predictor to get keypoint predictions
        with torch.no_grad():
            outputs = self.predictor_kp(frame)["instances"]

        # Visualize predictions on the frame
        v = DetectronVisualizer(
            frame[:, :, ::-1],
            MetadataCatalog.get(self.cfg_kp.DATASETS.TRAIN[0]),
            scale=1.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        output = v.draw_instance_predictions(outputs.to("cpu"))

        # Update frame with the visualized result
        out_frame = output.get_image()[:, :, ::-1]
        print("outframe is ready and returned - end of predict_keypoints")
        return out_frame, outputs

    def predict_densepose(self, frame):
        print("strating predict densepose")
        with torch.no_grad():
            outputs = self.predictor_dp(frame)["instances"]
        densepose_outputs = self.extractor(outputs)

        out_frame = frame.copy()
        out_frame_seg = np.zeros(out_frame.shape, dtype=out_frame.dtype)

        # Debugging outputs
        #print(f"Densepose outputs: {densepose_outputs}")
        print("densepose outputs ready and returing")
        # Ensure that densepose_outputs is not empty
        if not densepose_outputs:
            print("No densepose outputs detected.")
            return out_frame, out_frame_seg, densepose_outputs

        # Visualize DensePose results
        self.visualizer.visualize(out_frame, densepose_outputs)
        self.visualizer.visualize(out_frame_seg, densepose_outputs)

        return out_frame, out_frame_seg, densepose_outputs

    def save_keypoints(self, outputs):
        instances = outputs

        if hasattr(instances, 'pred_keypoints'):
            pred_keypoints = instances.pred_keypoints
            all_pred_keypoints = [keypoints.tolist() for keypoints in pred_keypoints]
            print(f'The keypoints are: {all_pred_keypoints}')
            return all_pred_keypoints
        else:
            print("The 'pred_keypoints' attribute is not present in the given Instances object.")

    def save_densepose(self, outputs):
        all_densepose = []
        for result in outputs:
            # Debugging outputs to inspect the structure
            #print(f"DensePose result: {result}")

            if isinstance(result, list):
                # Handle case where result is a list
                for res in result:
                    if hasattr(res, 'labels'):
                        densepose_result = {
                            "labels": res.labels.tolist(),
                            "uv": res.uv.tolist(),
                        }
                    elif isinstance(res, torch.Tensor):
                        # Handle the case where result is a tensor
                        densepose_result = res.tolist()
                    else:
                        print("Unknown result structure")
                        continue
                    all_densepose.append(densepose_result)
            else:
                # Handle the single DensePoseChartResultWithConfidences object
                if hasattr(result, 'labels'):
                    densepose_result = {
                        "labels": result.labels.tolist(),
                        "uv": result.uv.tolist(),
                    }
                elif isinstance(result, torch.Tensor):
                    # Handle the case where result is a tensor
                    densepose_result = result.tolist()
                else:
                    print("Unknown result structure")
                    continue
                all_densepose.append(densepose_result)
                
        return all_densepose

    def onImage(self, input_path, output_path):
        image = cv2.imread(input_path)

        keypoint_frame, keypoint_outputs = self.predict_keypoints(image)
        densepose_frame, densepose_seg, densepose_outputs = self.predict_densepose(image)

        cv2.imwrite(output_path + "_keypoints.jpg", keypoint_frame)
        cv2.imwrite(output_path + "_densepose.jpg", densepose_frame)

        keypoints = self.save_keypoints(keypoint_outputs)
        densepose = self.save_densepose(densepose_outputs)

        with open(output_path + "_keypoints.json", 'w') as f:
            json.dump(keypoints, f)
        with open(output_path + "_densepose.json", 'w') as f:
            json.dump(densepose, f)

        return keypoint_frame, densepose_frame, keypoints, densepose

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
        visualizer = VideoVisualizer(MetadataCatalog.get(self.cfg_kp.DATASETS.TRAIN[0]), ColorMode.IMAGE)

        positions_list = []

        done = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoint_frame, keypoint_outputs = self.predict_keypoints(frame)
            densepose_frame, densepose_seg, densepose_outputs = self.predict_densepose(frame)

            out_video.write(densepose_frame)

            frame_keypoints = self.save_keypoints(keypoint_outputs)
            frame_densepose = self.save_densepose(densepose_outputs)

            if frame_keypoints:
                predicted_position = find_position(frame_keypoints)
                positions_list.append({"frame_number": done, "keypoints": frame_keypoints, "position": predicted_position, "densepose": frame_densepose})
                print(f"Frame {done} - Position: {predicted_position}")
            else:
                print(f"Keypoints not available for frame {done}")

            done += 1

        cap.release()
        out_video.release()
        return positions_list
