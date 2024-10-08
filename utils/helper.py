from datetime import datetime, timedelta
import os
import cv2
import torch
import numpy as np
import json
import joblib
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .shared_utils import logger, Config, update_job_status, s3_client, BUCKET_NAME, get_job_details
from .find_position import find_position
import pyarrow as pa
import pyarrow.parquet as pq

class Predictor:
    def __init__(self):
        self.model = None
        try:
            self.model = joblib.load(Config.MODEL_PATH)
            logger.info(f"Model loaded successfully from {Config.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model from {Config.MODEL_PATH}: {str(e)}")

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
            return None, None

    def save_keypoints(self, outputs):
        try:
            instances = outputs
            if hasattr(instances, 'pred_keypoints'):
                pred_keypoints = instances.pred_keypoints
                all_pred_keypoints = [keypoints.tolist() for keypoints in pred_keypoints]
                return all_pred_keypoints
            else:
                logger.warning("The 'pred_keypoints' attribute is not present in the given Instances object.")
                return None
        except Exception as e:
            logger.error(f"Error in save_keypoints: {str(e)}")
            return None

    def onImage(self, input_path, output_path):
        try:
            if isinstance(input_path, np.ndarray):
                image = input_path
            else:
                image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image from {input_path}")

            keypoint_frame, keypoint_outputs = self.predict_keypoints(image)
            if keypoint_frame is None or keypoint_outputs is None:
                logger.error("Failed to predict keypoints")
                return None, None, None

            if isinstance(keypoint_frame, np.ndarray) and keypoint_frame.size > 0:
                cv2.imwrite(f"{output_path}_keypoints.jpg", keypoint_frame)
            else:
                logger.warning(f"Invalid keypoint_frame, not saving the image.")

            keypoints = self.save_keypoints(keypoint_outputs)
            if keypoints is None:
                logger.error("Failed to extract keypoints")
                return None, None, None

            with open(f"{output_path}_keypoints.json", 'w') as f:
                json.dump(keypoints, f)

            predicted_position = find_position(keypoints)

            return keypoint_frame, keypoints, predicted_position
        except Exception as e:
            logger.error(f"Error in onImage: {str(e)}")
            return None, None, None


            predicted_position = find_position(keypoints)

            return keypoint_frame, keypoints, predicted_position
        except Exception as e:
            logger.error(f"Error in onImage: {str(e)}")
            return None, None, None


class VideoProcessor:
    def __init__(self):
        self.predictor = Predictor()

    def process_video(self, video_path, output_path, job_id, user_id, progress_callback):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_skip = int(fps * 2.5)  # Process a frame every 2.5 seconds

            positions = []
            current_position = None
            start_time = None

            for frame_number in range(0, frame_count, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
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
                            'player_id': 1
                        })
                    current_position = predicted_position
                    start_time = timestamp

                progress_callback(frame_number)

            cap.release()

            if current_position is not None:
                positions.append({
                    'position': current_position,
                    'start_time': start_time,
                    'end_time': timedelta(seconds=frame_count / fps),
                    'player_id': 1
                })

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
            parquet_file = f'{output_path}/{job_id}.parquet'
            pq.write_table(table, parquet_file)
            logger.info(f"Parquet file written: {parquet_file}")

            return positions
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}")
            raise

        
def process_video_async(video_path, output_path, job_id, user_id):
    try:
        start_time = int(time.time())
        total_frames = get_total_frames(video_path)

        update_job_status(job_id, user_id, 'PROCESSING', 'video', video_path, 
                          processing_start_time=start_time,
                          total_frames=total_frames,
                          processed_frames=0)

        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

        video_processor = VideoProcessor()
        logger.info(f"Starting video processing for job_id: {job_id}")

        def progress_callback(frame_number):
            progress = (frame_number / total_frames) * 100
            update_job_status(job_id, user_id, 'PROCESSING', 'video', video_path,
                              progress=progress,
                              processed_frames=frame_number)

        positions = video_processor.process_video(video_path, output_path, job_id, user_id, progress_callback)
        logger.info(f"Video processing completed for job_id: {job_id}")

        job_details = get_job_details(job_id, user_id)
        submission_date = job_details.get('submission_date', datetime.utcnow().strftime('%Y-%m-%d'))

        s3_path = f'processed_data/user_id={user_id}/date={submission_date}/{job_id}.parquet'
        parquet_file = f'{output_path}/{job_id}.parquet'

        if os.path.exists(parquet_file):
            s3_client.upload_file(parquet_file, BUCKET_NAME, s3_path)
            logger.info(f"Uploaded {parquet_file} to S3: {s3_path}")
            
            os.remove(parquet_file)
            logger.info(f"Removed local parquet file: {parquet_file}")
        else:
            logger.error(f"Parquet file not found: {parquet_file}")
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

        end_time = int(time.time())
        update_job_status(job_id, user_id, 'COMPLETED', 'video', video_path, 
                          s3_path=s3_path, processing_end_time=end_time)
        logger.info(f"Updated job status to COMPLETED for job_id: {job_id}")

        return positions
    except Exception as e:
        logger.error(f"Error in process_video_async: {str(e)}", exc_info=True)
        update_job_status(job_id, user_id, 'FAILED', 'video', video_path)
        raise
    finally:
        for filename in os.listdir(output_path):
            if filename.startswith(f"frame_{job_id}"):
                os.remove(os.path.join(output_path, filename))
        logger.info(f"Cleaned up temporary files for job_id: {job_id}")


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames
# TODO: Implement unit tests for Predictor, VideoProcessor, and process_video_async functions