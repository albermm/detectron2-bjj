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
from .find_position import find_position

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

    def predict_objects(self, frame):
        try:
            with torch.no_grad():
                outputs = self.predictor_kp(frame)
            return outputs
        except Exception as e:
            logger.error(f"Error in predict_objects: {str(e)}")
            return None

    def save_detection_data(self, keypoint_outputs, object_outputs):
        try:
            if hasattr(keypoint_outputs, 'pred_keypoints') and hasattr(object_outputs, 'pred_boxes'):
                pred_keypoints = keypoint_outputs.pred_keypoints
                pred_boxes = object_outputs.pred_boxes
                all_pred_keypoints = [keypoints.tolist() for keypoints in pred_keypoints]
                all_pred_boxes = [box.tensor.tolist() for box in pred_boxes]
                return all_pred_keypoints, all_pred_boxes
            else:
                logger.warning("Required attributes not present in the given outputs.")
                return None, None
        except Exception as e:
            logger.error(f"Error in save_detection_data: {str(e)}")
            return None, None

    def on_image(self, input_path, output_path):
        try:
            if isinstance(input_path, np.ndarray):
                image = input_path
            else:
                image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image from {input_path}")

            keypoint_frame, keypoint_outputs = self.predict_keypoints(image)
            object_outputs = self.predict_objects(image)

            if keypoint_frame is None or keypoint_outputs is None or object_outputs is None:
                logger.error("Failed to predict keypoints or objects")
                return None, None, None, None

            if isinstance(keypoint_frame, np.ndarray) and keypoint_frame.size > 0:
                cv2.imwrite(f"{output_path}_keypoints.jpg", keypoint_frame)
            else:
                logger.warning(f"Invalid keypoint_frame, not saving the image.")

            keypoints, bounding_boxes = self.save_detection_data(keypoint_outputs, object_outputs)
            if keypoints is None or bounding_boxes is None:
                logger.error("Failed to extract keypoints or bounding boxes")
                return None, None, None, None

            # Save both keypoints and bounding boxes to JSON
            with open(f"{output_path}_detection_data.json", 'w') as f:
                json.dump({'keypoints': keypoints, 'bounding_boxes': bounding_boxes}, f)

            predicted_position = find_position(keypoints)

            return keypoint_frame, keypoints, predicted_position, bounding_boxes
        except Exception as e:
            logger.error(f"Error in on_image: {str(e)}")
            return None, None, None, None

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

            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            pose_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]

            for frame_number in range(0, frame_count, frame_skip):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_number}")
                    break

                timestamp = timedelta(seconds=frame_number / fps)

                keypoint_frame, keypoints, _, object_outputs = self.predictor.on_image(frame, f"{output_path}/frame_{frame_number}")
                

                if keypoints and bounding_boxes and len(keypoints) == len(bounding_boxes):
                    updated_keypoints, updated_boxes = self.tracker.update(frame, keypoints, bounding_boxes)
                else:
                    logger.warning(f"Invalid detections in frame {frame_number}")
                    continue

                # Interaction Detection
                interactions = self.detect_interactions(updated_boxes)

                for player_id, (keypoint, box) in enumerate(zip(updated_keypoints, updated_boxes)):
                    color = colors[player_id % len(colors)]
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = box[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Calculate keypoint quality and handle occlusions
                    keypoint_quality = self.calculate_keypoint_quality(np.array(keypoint))
                    occluded = self.detect_occlusion(keypoint, updated_boxes)
                    
                    # Find and smooth position
                    position, confidence = self.tracker.find_position([keypoint])
                    if player_id not in self.position_smoothers:
                        self.position_smoothers[player_id] = PositionSmoother()
                    smoothed_position, smoothed_confidence = self.position_smoothers[player_id].update(position, confidence)
                    
                    # Draw player ID, position, and occlusion status
                    status_text = f"Player {player_id}: {smoothed_position}"
                    if occluded:
                        status_text += " (Occluded)"
                    cv2.putText(frame, status_text, (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw keypoints and connections
                    self.draw_keypoints_and_connections(frame, keypoint, color, pose_pairs)

                    # Update position data
                    if player_id not in current_positions or smoothed_position != current_positions[player_id]:
                        if player_id in current_positions:
                            positions.append({
                                'position': smoothed_position,
                                'start_time': start_times[player_id],
                                'end_time': timestamp,
                                'player_id': player_id,
                                'confidence': smoothed_confidence,
                                'keypoint_quality': keypoint_quality,
                                'is_smoothed': True,
                                'bounding_box': box[0].tolist(), 
                                'keypoints': keypoint,
                                'occluded': occluded
                            })
                        current_positions[player_id] = smoothed_position
                        start_times[player_id] = timestamp

                # Draw interaction lines
                self.draw_interactions(frame, interactions, updated_boxes)

                out.write(frame)
                progress_callback(frame_number / frame_count * 100)

            # Add final positions
            for player_id, position in current_positions.items():
                if player_id < len(updated_keypoints):
                    keypoint_quality = self.calculate_keypoint_quality(np.array(updated_keypoints[player_id]))
                    smoothed_position, smoothed_confidence = self.position_smoothers[player_id].update(position, confidence)
                    occluded = self.detect_occlusion(updated_keypoints[player_id], updated_boxes)
                    positions.append({
                        'position': smoothed_position,
                        'start_time': start_times[player_id],
                        'end_time': timedelta(seconds=frame_count / fps),
                        'player_id': player_id,
                        'confidence': smoothed_confidence,
                        'keypoint_quality': keypoint_quality,
                        'is_smoothed': True,
                        'bounding_box': updated_boxes[player_id][0].tolist(),
                        'keypoints': updated_keypoints[player_id],
                        'occluded': occluded
                    })

            logger.info(f"Video processing completed. Total positions detected: {len(positions)}")
            return positions, processed_video_path

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

    def detect_interactions(self, bounding_boxes: List[List[float]]) -> List[Tuple[int, int]]:
        interactions = []
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                if self.boxes_overlap(bounding_boxes[i][0], bounding_boxes[j][0]):
                    interactions.append((i, j))
        return interactions

    def boxes_overlap(self, box1: List[float], box2: List[float]) -> bool:
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

    def detect_occlusion(self, keypoints: List[float], all_boxes: List[List[float]]) -> bool:
        keypoint_array = np.array(keypoints).reshape(-1, 3)
        visible_keypoints = keypoint_array[keypoint_array[:, 2] > 0]
        if len(visible_keypoints) / len(keypoint_array) < 0.5:  # If less than 50% of keypoints are visible
            return True
        return False

    def draw_keypoints_and_connections(self, frame: np.ndarray, keypoint: List[float], color: Tuple[int, int, int], pose_pairs: List[Tuple[int, int]]):
        keypoints_array = np.array(keypoint).reshape(-1, 3)
        for kp in keypoints_array:
            if kp[2] > 0:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)
        
        for pair in pose_pairs:
            if keypoints_array[pair[0], 2] > 0 and keypoints_array[pair[1], 2] > 0:
                pt1 = (int(keypoints_array[pair[0], 0]), int(keypoints_array[pair[0], 1]))
                pt2 = (int(keypoints_array[pair[1], 0]), int(keypoints_array[pair[1], 1]))
                cv2.line(frame, pt1, pt2, color, 2)

    def draw_interactions(self, frame: np.ndarray, interactions: List[Tuple[int, int]], bounding_boxes: List[List[float]]):
        for i, j in interactions:
            center1 = self.get_box_center(bounding_boxes[i][0])
            center2 = self.get_box_center(bounding_boxes[j][0])
            cv2.line(frame, center1, center2, (0, 0, 255), 2)

    def get_box_center(self, box: List[float]) -> Tuple[int, int]:
        x1, y1, x2, y2 = map(int, box)
        return ((x1 + x2) // 2, (y1 + y2) // 2)



def generate_s3_path(user_id, job_id, file_type):
    user_hash = hashlib.md5(user_id.encode()).hexdigest()
    now = datetime.utcnow()
    year, month, day = now.strftime("%Y/%m/%d").split('/')
    extension = '.parquet' if file_type == 'data' else '.mp4'
    s3_path = f'processed_{file_type}/{user_hash}/{year}/{month}/{day}/{job_id}{extension}'
    return s3_path


#initiates the video processing task in an asynchronous manner. It's designed to: Set up the necessary parameters and environment 
#for video processing. Start the actual video processing in a separate thread. Update the job status in the database as the processing progresses.
#Handle any errors that occur during processing.
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

        video_processor = VideoProcessor(frame_interval=frame_interval)
        logger.info(f"Starting video processing for job_id: {job_id}")

        def progress_callback(progress):
            processed_frames = int((progress / 100) * total_frames)
            update_job_status(job_id, user_id, 'PROCESSING', 'video', video_path,
                              progress=progress,
                              processed_frames=processed_frames)

        positions, processed_video_path = video_processor.process_video(video_path, output_path, job_id, user_id, progress_callback)
        logger.info(f"Video processing completed for job_id: {job_id}")

        # Upload processed video to S3
        processed_video_s3_path = generate_s3_path(user_id, job_id, 'video')
        s3_client.upload_file(processed_video_path, BUCKET_NAME, processed_video_s3_path)
        logger.info(f"Uploaded processed video to S3: {processed_video_s3_path}")

        # Save positions data to parquet and upload to S3
        data_s3_path = generate_s3_path(user_id, job_id, 'data')
        parquet_file = f'{output_path}/{job_id}.parquet'
        temp_files.extend([parquet_file, processed_video_path])
        
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
            'is_smoothed': [pos['is_smoothed'] for pos in positions],
            'bounding_box_x1': [pos['bounding_box'][0] for pos in positions],
            'bounding_box_y1': [pos['bounding_box'][1] for pos in positions],
            'bounding_box_x2': [pos['bounding_box'][2] for pos in positions],
            'bounding_box_y2': [pos['bounding_box'][3] for pos in positions],
            'keypoints': [json.dumps(pos['keypoints']) for pos in positions] 
        }
        df = pa.Table.from_pydict(data)
        pq.write_table(df, parquet_file)

        s3_client.upload_file(parquet_file, BUCKET_NAME, data_s3_path)
        logger.info(f"Uploaded {parquet_file} to S3: {data_s3_path}")

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
        logger.info(f"Cleaned up {len(frame_files)} temporary frame files for job_id: {job_id}")

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