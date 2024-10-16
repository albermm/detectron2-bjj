import cv2
import torch
import numpy as np
import json
import joblib
import os
import time
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from collections import deque
from detectron2.utils.visualizer import Visualizer as DetectronVisualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from .shared_utils import (
    logger, Config, update_job_status, s3_client, BUCKET_NAME, 
    get_job_details, dynamodb_table
)
from .find_position import find_position
from .combined_tracker import CombinedTracker

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

class PositionSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)

    def update(self, position: str, confidence: float) -> Tuple[str, float]:
        self.position_history.append(position)
        self.confidence_history.append(confidence)

        if len(self.position_history) < self.window_size:
            return position, confidence

        # Calculate the most common position
        smoothed_position = max(set(self.position_history), key=self.position_history.count)

        # Calculate the weighted average confidence
        total_weight = sum(self.confidence_history)
        smoothed_confidence = sum(c * w for c, w in zip(self.confidence_history, self.confidence_history)) / total_weight

        return smoothed_position, smoothed_confidence

class VideoProcessor:
    def __init__(self, frame_interval: float = 0.1, position_change_threshold: float = 0.1):
        self.predictor = Predictor()
        self.tracker = CombinedTracker()
        self.frame_interval = frame_interval
        self.position_change_threshold = position_change_threshold
        self.position_smoothers = {}

    def process_video(self, video_path: str, output_path: str, job_id: str, user_id: str, progress_callback: callable) -> Tuple[List[Dict], str]:
        cap = None
        out = None
        positions: List[Dict] = []
        processed_video_path = ""
        current_positions = {}
        start_times = {}

        try:
            if not hasattr(self, 'predictor') or not hasattr(self, 'tracker'):
                raise AttributeError("Predictor or tracker not initialized")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_skip = max(1, int(fps * self.frame_interval))

            processed_video_path = os.path.join(output_path, f"{job_id}_processed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

            for frame_number in range(0, frame_count, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_number}")
                    break

                timestamp = timedelta(seconds=frame_number / fps)

                _, keypoints, _ = self.predictor.onImage(frame, f"{output_path}/frame_{frame_number}")
                logger.debug(f"Frame {frame_number}: Detected {len(keypoints)} keypoints")

                if keypoints and isinstance(keypoints[0], (list, np.ndarray)) and len(keypoints[0]) > 0:
                    updated_keypoints = self.tracker.update(frame, keypoints)
                else:
                    logger.warning(f"Invalid keypoints detected in frame {frame_number}")
                    continue

                all_pred_keypoints = updated_keypoints
                predicted_position = self.tracker.find_position(all_pred_keypoints)

                for player_id, keypoint in enumerate(updated_keypoints):
                    keypoint_quality = self.calculate_keypoint_quality(np.array(keypoint))
                    position, confidence = self.tracker.find_position([keypoint])
                    
                    if player_id not in self.position_smoothers:
                        self.position_smoothers[player_id] = PositionSmoother()
                    
                    smoothed_position, smoothed_confidence = self.position_smoothers[player_id].update(position, confidence)
                    
                    if player_id not in current_positions or smoothed_position != current_positions[player_id]:
                        if player_id in current_positions:
                            positions.append({
                                'position': current_positions[player_id],
                                'start_time': start_times[player_id],
                                'end_time': timestamp,
                                'player_id': player_id,
                                'confidence': smoothed_confidence,
                                'keypoint_quality': keypoint_quality,
                                'is_smoothed': True
                            })
                        current_positions[player_id] = smoothed_position
                        start_times[player_id] = timestamp

                # Draw bounding boxes for visualization
                for player_id, keypoint in enumerate(updated_keypoints):
                    box = self.tracker.keypoint_to_box(keypoint)
                    if box:
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Player {player_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                out.write(frame)

                progress_callback(frame_number / frame_count * 100)  # Report progress as percentage

            # Add final positions
            for player_id, position in current_positions.items():
                keypoint_quality = self.calculate_keypoint_quality(np.array(updated_keypoints[player_id]))
                smoothed_position, smoothed_confidence = self.position_smoothers[player_id].update(position, confidence)
                positions.append({
                    'position': smoothed_position,
                    'start_time': start_times[player_id],
                    'end_time': timedelta(seconds=frame_count / fps),
                    'player_id': player_id,
                    'confidence': smoothed_confidence,
                    'keypoint_quality': keypoint_quality,
                    'is_smoothed': True
                })

            logger.info(f"Video processing completed. Total positions detected: {len(positions)}")
            return positions, processed_video_path

        except cv2.error as e:
            logger.error(f"OpenCV error in process_video: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}", exc_info=True)
            raise

        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    def calculate_keypoint_quality(self, keypoints: np.ndarray) -> float:
        if keypoints.size == 0:
            return 0.0
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(1, -1)
        if keypoints.shape[1] % 3 != 0:
            logger.warning(f"Unexpected keypoint shape: {keypoints.shape}")
            return 0.0
        keypoints = keypoints.reshape(-1, 3)
        valid_keypoints = keypoints[keypoints[:, 2] > 0]
        return len(valid_keypoints) / len(keypoints)


def generate_s3_path(user_id, job_id, file_type):
    user_hash = hashlib.md5(user_id.encode()).hexdigest()
    now = datetime.utcnow()
    year, month, day = now.strftime("%Y/%m/%d").split('/')
    extension = '.parquet' if file_type == 'data' else '.mp4'
    s3_path = f'processed_{file_type}/{user_hash}/{year}/{month}/{day}/{job_id}{extension}'
    return s3_path

def process_video_async(video_path, output_path, job_id, user_id, frame_interval=2.5):
    start_time = int(time.time())
    temp_files = []

    try:
        total_frames = get_total_frames(video_path)
        if total_frames == 0:
            raise ValueError(f"Invalid video file or no frames detected: {video_path}")

        update_job_status(job_id, user_id, 'PROCESSING', 'video', video_path, 
                          processing_start_time=start_time,
                          total_frames=total_frames,
                          processed_frames=0)

        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

        video_processor = VideoProcessor(frame_interval=2.5)
        logger.info(f"Starting video processing for job_id: {job_id}")

        def progress_callback(frame_number):
            progress = round(min(100, max(0, (frame_number / total_frames) * 100)), 0)
            update_job_status(job_id, user_id, 'PROCESSING', 'video', video_path,
                              progress=progress,
                              processed_frames=frame_number)

        job_id = str(job_id)
        user_id = str(user_id)

        positions, processed_video_path = video_processor.process_video(video_path, output_path, job_id, user_id, progress_callback)
        logger.info(f"Video processing completed for job_id: {job_id}")

        processed_video_s3_path = generate_s3_path(user_id, job_id, 'video')
        s3_client.upload_file(processed_video_path, BUCKET_NAME, processed_video_s3_path)
        logger.info(f"Uploaded processed video to S3: {processed_video_s3_path}")

        data_s3_path = generate_s3_path(user_id, job_id, 'data')
        parquet_file = f'{output_path}/{job_id}.parquet'
        temp_files.append(parquet_file)
        temp_files.append(processed_video_path)
        
        # Create and save parquet file
        data = {
            'job_id': [job_id] * len(positions),
            'user_id': [user_id] * len(positions),
            'player_id': [pos['player_id'] for pos in positions],
            'position': [pos['position'] for pos in positions],
            'start_time': [pos['start_time'].total_seconds() for pos in positions],
            'end_time': [pos['end_time'].total_seconds() for pos in positions],
            'duration': [(pos['end_time'] - pos['start_time']).total_seconds() for pos in positions],
            'video_timestamp': [pos['start_time'].total_seconds() for pos in positions],
            'confidence': [pos['confidence'] for pos in positions],
            'keypoint_quality': [pos['keypoint_quality'] for pos in positions],
            'is_smoothed': [pos['is_smoothed'] for pos in positions]
        }
        df = pa.Table.from_pydict(data)
        pq.write_table(df, parquet_file)

        if os.path.exists(parquet_file):
            s3_client.upload_file(parquet_file, BUCKET_NAME, data_s3_path)
            logger.info(f"Uploaded {parquet_file} to S3: {data_s3_path}")
        else:
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

        update_job_status(job_id, user_id, 'COMPLETED', 'video', video_path, 
                          s3_path=data_s3_path,
                          processed_video_s3_path=processed_video_s3_path,
                          processing_end_time=int(time.time()),
                          total_frames=total_frames,
                          processed_frames=total_frames)

        logger.info(f"Updated job status to COMPLETED for job_id: {job_id}")

        return positions

    except Exception as e:
        logger.error(f"Error in process_video_async: {str(e)}", exc_info=True)
        update_job_status(job_id, user_id, 'FAILED', 'video', video_path)
        raise

    finally:
        # Clean up temporary files
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed temporary file: {file}")

        # Clean up frame files
        frame_files = [f for f in os.listdir(output_path) if f.startswith(f"frame_{job_id}")]
        for file in frame_files:
            os.remove(os.path.join(output_path, file))
        logger.info(f"Cleaned up temporary files for job_id: {job_id}")

def get_total_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames
    except Exception as e:
        logger.error(f"Error in get_total_frames: {str(e)}")
        return 0
    finally:
        if cap:
            cap.release()

# TODO: Implement unit tests for Predictor, VideoProcessor, and process_video_async functions