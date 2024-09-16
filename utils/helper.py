import cv2
import torch
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .shared_utils import logger, s3_client, BUCKET_NAME, update_job_status, Config
from .find_position import find_position

class Predictor:
    def __init__(self):
        self.cfg_kp = get_cfg()
        self.cfg_kp.merge_from_file(model_zoo.get_config_file(Config.KEYPOINT_CONFIG))
        self.cfg_kp.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(Config.KEYPOINT_CONFIG)
        self.cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config.KEYPOINT_THRESHOLD
        self.cfg_kp.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_kp = DefaultPredictor(self.cfg_kp)

    def predict_keypoints(self, frame):
        try:
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
            return out_frame, outputs
        except Exception as e:
            logger.error(f"Error in predict_keypoints: {str(e)}")
            raise

  
    def save_keypoints(self, outputs):
        try:
            instances = outputs
            if hasattr(instances, 'pred_keypoints'):
                pred_keypoints = instances.pred_keypoints
                all_pred_keypoints = [keypoints.cpu().numpy().tolist() for keypoints in pred_keypoints]
                return all_pred_keypoints
            else:
                logger.warning("The 'pred_keypoints' attribute is not present in the given Instances object.")
                return None
        except Exception as e:
            logger.error(f"Error in save_keypoints: {str(e)}")
            raise

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
            logger.error(f"Error in onImage: {str(e)}")
            return None, None, None

class VideoProcessor:
    def __init__(self):
        self.predictor = Predictor()

    def process_video(self, video_path, output_path, job_id, user_id):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            positions = []
            current_position = None
            start_time = None

            for frame_number in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = timedelta(seconds=frame_number / fps)
                
                _, keypoints, predicted_position = self.predictor.onImage(frame, f"{output_path}/frame_{frame_number}")

                if predicted_position != current_position:
                    if current_position is not None:
                        positions.append({
                            'position': current_position,
                            'start_time': start_time,
                            'end_time': timestamp,
                            'player_id': 1  # Assuming single player for simplicity
                        })
                    current_position = predicted_position
                    start_time = timestamp

                if frame_number % 100 == 0:
                    progress = (frame_number + 1) / frame_count * 100
                    update_job_status(job_id, user_id, f"PROCESSING - {progress:.2f}%", 'video', video_path)

            cap.release()

            # Add the last position
            if current_position is not None:
                positions.append({
                    'position': current_position,
                    'start_time': start_time,
                    'end_time': timedelta(seconds=frame_count / fps),
                    'player_id': 1
                })

            # Convert positions to Parquet
            data = {
                'job_id': [job_id] * len(positions),
                'user_id': [user_id] * len(positions),
                'player_id': [pos['player_id'] for pos in positions],
                'position': [pos['position'] for pos in positions],
                'start_time': [pos['start_time'].total_seconds() for pos in positions],
                'end_time': [pos['end_time'].total_seconds() for pos in positions],
                'duration': [(pos['end_time'] - pos['start_time']).total_seconds() for pos in positions],
                'video_timestamp': [pos['start_time'].total_seconds() for pos in positions]
            }

            table = pa.Table.from_pydict(data)
            pq.write_table(table, f'{output_path}/{job_id}.parquet')

            return positions
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            raise
        
def process_video_async(video_path, output_path, job_id, user_id):
    try:
        video_processor = VideoProcessor()
        positions = video_processor.process_video(video_path, output_path, job_id, user_id)
        
        # Upload to S3
        current_date = datetime.now().strftime('%Y-%m-%d')
        s3_path = f'processed_data/user_id={user_id}/date={current_date}/{job_id}.parquet'
        s3_client.upload_file(f'{output_path}/{job_id}.parquet', BUCKET_NAME, s3_path)

        # Update DynamoDB with job completion status
        update_job_status(job_id, user_id, 'COMPLETED', 'video', video_path, s3_path=s3_path)

        return positions
    except Exception as e:
        logger.error(f"Error in process_video_async: {str(e)}")
        update_job_status(job_id, user_id, 'FAILED', 'video', video_path)
        raise

# TODO: Implement unit tests for Predictor, VideoProcessor, and process_video_async functions