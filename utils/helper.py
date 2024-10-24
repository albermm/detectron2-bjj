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

        # Keypoint detection configuration
        self.cfg_kp = get_cfg()
        self.cfg_kp.merge_from_file(model_zoo.get_config_file(Config.KEYPOINT_CONFIG))
        self.cfg_kp.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(Config.KEYPOINT_CONFIG)
        self.cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config.KEYPOINT_THRESHOLD
        self.cfg_kp.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_kp = DefaultPredictor(self.cfg_kp)
        logger.info(f"Keypoint model configuration: {self.cfg_kp}")

        # Object detection configuration
        self.cfg_obj = get_cfg()
        self.cfg_obj.merge_from_file(model_zoo.get_config_file(Config.BOUNDING_BOX_CONFIG))
        self.cfg_obj.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(Config.BOUNDING_BOX_CONFIG)
        self.cfg_obj.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config.BOX_THRESHOLD
        self.cfg_obj.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor_obj = DefaultPredictor(self.cfg_obj)
        logger.info(f"Object detection model configuration: {self.cfg_obj}")

    def debug_instances(self, instances, source="unknown"):
        try:
            logger.info(f"Debugging instances from {source}:")
            logger.info(f"Type: {type(instances)}")
            logger.info(f"Available fields: {instances.get_fields() if hasattr(instances, 'get_fields') else 'No fields method'}")
            
            if hasattr(instances, '_fields'):
                logger.info(f"Raw fields: {instances._fields}")
            
            for field in ['pred_boxes', 'pred_keypoints', 'scores']:
                if hasattr(instances, field):
                    attr = getattr(instances, field)
                    logger.info(f"{field} shape: {attr.shape if hasattr(attr, 'shape') else 'No shape'}")
                    logger.info(f"{field} type: {type(attr)}")
        except Exception as e:
            logger.error(f"Error in debug_instances: {str(e)}", exc_info=True)


    def predict_keypoints(self, frame):
        try:
            logger.info("Starting keypoint prediction")
            logger.info(f"Input frame shape: {frame.shape}")
            
            with torch.no_grad():
                outputs = self.predictor_kp(frame)
                logger.info("Raw predictor output keys: " + str(outputs.keys()))
            
            instances = outputs["instances"]
            self.debug_instances(instances, "keypoint_prediction")
            
            v = DetectronVisualizer(
                frame[:, :, ::-1],
                MetadataCatalog.get(self.cfg_kp.DATASETS.TRAIN[0]),
                scale=1.5,
                instance_mode=ColorMode.IMAGE_BW,
            )
            output = v.draw_instance_predictions(instances.to("cpu"))
            out_frame = output.get_image()[:, :, ::-1]
            
            logger.info("Keypoint prediction completed successfully")
            return out_frame, instances
        except Exception as e:
            logger.error(f"Error in predict_keypoints: {str(e)}", exc_info=True)
            return None, None

    def predict_objects(self, frame):
        try:
            logger.info("Starting object detection")
            with torch.no_grad():
                outputs = self.predictor_obj(frame)
            instances = outputs["instances"]
            self.debug_instances(instances, "object_detection")
            
            logger.info("Object detection completed successfully")
            return instances
        except Exception as e:
            logger.error(f"Error in predict_objects: {str(e)}", exc_info=True)
            return None

    def visualize_detections(self, frame: np.ndarray, keypoints: List[List[float]], 
                        bounding_boxes: List[List[float]], positions: Dict[int, str] = None) -> np.ndarray:
        """
        Visualize detections with configurable confidence threshold
        """
        frame_copy = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        keypoint_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), 
                        (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]

        logger.debug(f"Visualizing: {len(keypoints)} keypoints, {len(bounding_boxes)} boxes")

        # Draw bounding boxes first
        for idx, box_info in enumerate(bounding_boxes):
            try:
                color = colors[idx % len(colors)]
                # Handle different box formats
                if isinstance(box_info, list):
                    box = box_info[0] if isinstance(box_info[0], list) else box_info
                else:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

                # Add position label if available
                if positions and idx in positions:
                    label = f"Player {idx + 1}: {positions[idx]}"
                    cv2.putText(frame_copy, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                logger.warning(f"Error drawing bounding box {idx}: {str(e)}")
                continue

        # Draw keypoints and connections using lower threshold
        for idx, person_keypoints in enumerate(keypoints):
            try:
                color = colors[idx % len(colors)]
                keypoint_array = np.array(person_keypoints).reshape(-1, 3)

                # Draw connections
                for pair in keypoint_pairs:
                    if (pair[0] < len(keypoint_array) and 
                        pair[1] < len(keypoint_array) and
                        keypoint_array[pair[0], 2] > Config.KEYPOINT_VIZ_THRESHOLD and 
                        keypoint_array[pair[1], 2] > Config.KEYPOINT_VIZ_THRESHOLD):
                        
                        pt1 = tuple(map(int, keypoint_array[pair[0], :2]))
                        pt2 = tuple(map(int, keypoint_array[pair[1], :2]))
                        cv2.line(frame_copy, pt1, pt2, color, 2)

                # Draw keypoints
                for kp in keypoint_array:
                    if kp[2] > Config.KEYPOINT_VIZ_THRESHOLD:  # Use visualization threshold
                        cv2.circle(frame_copy, 
                                (int(kp[0]), int(kp[1])), 
                                4, color, -1)

            except Exception as e:
                logger.warning(f"Error drawing keypoints for person {idx}: {str(e)}")
                continue

        return frame_copy

    def save_detection_data(self, keypoint_outputs, object_outputs):
        try:
            logger.info("Starting detection data extraction")
            self.debug_instances(keypoint_outputs, "keypoint_outputs_in_save")
            self.debug_instances(object_outputs, "object_outputs_in_save")
            
            all_pred_keypoints = []
            all_pred_boxes = []
            
            # Extract keypoints with detailed error checking
            if hasattr(keypoint_outputs, 'pred_keypoints'):
                pred_keypoints = keypoint_outputs.pred_keypoints
                logger.info(f"Found pred_keypoints with shape: {pred_keypoints.shape}")
                try:
                    all_pred_keypoints = pred_keypoints.cpu().numpy().tolist()
                    logger.info(f"Successfully extracted {len(all_pred_keypoints)} keypoint sets")
                except Exception as e:
                    logger.error(f"Error converting keypoints: {str(e)}", exc_info=True)
            else:
                logger.warning("pred_keypoints attribute not found in keypoint_outputs")
            
            # Extract bounding boxes with detailed error checking
            if hasattr(object_outputs, 'pred_boxes'):
                pred_boxes = object_outputs.pred_boxes
                if hasattr(pred_boxes, 'tensor'):
                    try:
                        # Get the tensor directly from the Boxes object
                        boxes_tensor = pred_boxes.tensor
                        boxes_np = boxes_tensor.cpu().numpy()
                        scores_np = object_outputs.scores.cpu().numpy()
                        
                        # Zip boxes with their confidence scores
                        all_pred_boxes = [[box.tolist(), float(score)] 
                                        for box, score in zip(boxes_np, scores_np)]
                        logger.info(f"Successfully extracted {len(all_pred_boxes)} bounding boxes")
                    except Exception as e:
                        logger.error(f"Error converting bounding boxes: {str(e)}", exc_info=True)
                else:
                    logger.warning("pred_boxes object does not have tensor attribute")
            else:
                logger.warning("pred_boxes attribute not found in object_outputs")
            
            return all_pred_keypoints, all_pred_boxes
        except Exception as e:
            logger.error(f"Error in save_detection_data: {str(e)}", exc_info=True)
            return [], []

    def filter_low_confidence_detections(self, keypoints, bounding_boxes, 
                                      keypoint_threshold=Config.KEYPOINT_VIZ_THRESHOLD,
                                      box_threshold=Config.BOX_THRESHOLD):
        """
        Filter detections based on confidence with lower thresholds
        """
        try:
            # Filter keypoints
            filtered_keypoints = []
            for person_keypoints in keypoints:
                quality = self.calculate_keypoint_quality(person_keypoints)
                if quality >= Config.MIN_KEYPOINT_QUALITY:
                    # Keep keypoints even with lower confidence for visualization
                    confident_keypoints = [kp for kp in person_keypoints if kp[2] > keypoint_threshold]
                    if confident_keypoints:
                        filtered_keypoints.append(person_keypoints)
            
            # Filter bounding boxes
            filtered_boxes = [box for box in bounding_boxes if box[1] > box_threshold]
            
            logger.info(f"Filtered keypoints: {len(filtered_keypoints)} from {len(keypoints)}")
            logger.info(f"Filtered boxes: {len(filtered_boxes)} from {len(bounding_boxes)}")
            
            return filtered_keypoints, filtered_boxes
        except Exception as e:
            logger.error(f"Error in filter_low_confidence_detections: {str(e)}")
            return keypoints, bounding_boxes
    

    def on_image(self, input_path, output_path):
        try:
            logger.info(f"Starting image processing: {input_path}")
            if isinstance(input_path, np.ndarray):
                image = input_path
            else:
                image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image from {input_path}")
            
            logger.info(f"Image loaded successfully. Shape: {image.shape}")
            
            # Predict keypoints and objects
            logger.info("Starting keypoint prediction")
            keypoint_frame, keypoint_outputs = self.predict_keypoints(image)
            
            logger.info("Starting object detection")
            object_outputs = self.predict_objects(image)
            
            if keypoint_frame is None or keypoint_outputs is None or object_outputs is None:
                logger.error("Failed to predict keypoints or objects")
                return None, None, None, None
            
            # Save visualization
            if isinstance(keypoint_frame, np.ndarray) and keypoint_frame.size > 0:
                cv2.imwrite(f"{output_path}_keypoints.jpg", keypoint_frame)
                logger.info(f"Saved keypoint visualization")
            
            # Extract and filter detections
            keypoints, bounding_boxes = self.save_detection_data(keypoint_outputs, object_outputs)
            if keypoints is None or bounding_boxes is None:
                logger.error("Failed to extract keypoints or bounding boxes")
                return None, None, None, None
            
            # Filter low confidence detections
            keypoints, bounding_boxes = self.filter_low_confidence_detections(keypoints, bounding_boxes)
            
            # Create and save combined visualization
            visualized_image = self.visualize_detections(image, keypoints, bounding_boxes)
            cv2.imwrite(f"{output_path}_visualized.jpg", visualized_image)
            logger.info(f"Saved combined visualization")
            
            # Save detection data
            with open(f"{output_path}_detection_data.json", 'w') as f:
                json.dump({
                    'keypoints': keypoints,
                    'bounding_boxes': bounding_boxes,
                    'num_keypoints': len(keypoints),
                    'num_boxes': len(bounding_boxes)
                }, f, indent=2)
            logger.info(f"Saved detection data")
            
            # Predict position
            predicted_position = find_position(keypoints)
            logger.info(f"Predicted position: {predicted_position}")
            
            return keypoint_frame, keypoints, predicted_position, bounding_boxes
            
        except Exception as e:
            logger.error(f"Error in on_image: {str(e)}", exc_info=True)
            return None, None, None, None



class VideoProcessor:
    def __init__(self, frame_interval: float = 0.1, position_change_threshold: float = 0.1):
        self.predictor = Predictor()
        self.tracker = CombinedTracker()
        self.frame_interval = frame_interval
        self.position_change_threshold = position_change_threshold
        self.position_smoothers = {}

    def _format_bounding_boxes(self, raw_boxes):
        """Convert raw bounding boxes to the expected format"""
        formatted_boxes = []
        for box_info in raw_boxes:
            if isinstance(box_info, list) and len(box_info) > 0:
                box = box_info[0] if isinstance(box_info[0], list) else box_info
                formatted_boxes.append(box)
        return formatted_boxes
        
    def visualize_detections(self, frame: np.ndarray, 
                           keypoints: List[List[float]], 
                           bounding_boxes: List[List[float]], 
                           positions: Dict[int, str] = None) -> np.ndarray:
        """
        Comprehensive visualization function that handles all detection types
        """
        frame_copy = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        keypoint_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), 
                         (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]

        logger.debug(f"Visualizing: {len(keypoints)} keypoints, {len(bounding_boxes)} boxes")

        # Draw bounding boxes first
        for idx, box_info in enumerate(bounding_boxes):
            try:
                color = colors[idx % len(colors)]
                # Handle different box formats
                if isinstance(box_info, list):
                    box = box_info[0] if isinstance(box_info[0], list) else box_info
                else:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

                # Add position label if available
                if positions and idx in positions:
                    label = f"Player {idx + 1}: {positions[idx]}"
                    cv2.putText(frame_copy, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                logger.warning(f"Error drawing bounding box {idx}: {str(e)}")
                continue

        # Draw keypoints and connections
        for idx, person_keypoints in enumerate(keypoints):
            try:
                color = colors[idx % len(colors)]
                keypoint_array = np.array(person_keypoints).reshape(-1, 3)

                # Draw connections
                for pair in keypoint_pairs:
                    if (pair[0] < len(keypoint_array) and 
                        pair[1] < len(keypoint_array) and
                        keypoint_array[pair[0], 2] > 0.5 and 
                        keypoint_array[pair[1], 2] > 0.5):
                        
                        pt1 = tuple(map(int, keypoint_array[pair[0], :2]))
                        pt2 = tuple(map(int, keypoint_array[pair[1], :2]))
                        cv2.line(frame_copy, pt1, pt2, color, 2)

                # Draw keypoints
                for kp_idx, kp in enumerate(keypoint_array):
                    if kp[2] > 0.5:  # Only draw high confidence keypoints
                        cv2.circle(frame_copy, 
                                 (int(kp[0]), int(kp[1])), 
                                 4, color, -1)
                        
                        # Add keypoint labels for debugging if needed
                        # cv2.putText(frame_copy, str(kp_idx), 
                        #           (int(kp[0]), int(kp[1])), 
                        #           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            except Exception as e:
                logger.warning(f"Error drawing keypoints for person {idx}: {str(e)}")
                continue

        return frame_copy

    def process_video(self, video_path: str, output_path: str, job_id: str, user_id: str, 
                     progress_callback: callable) -> Tuple[List[Dict], str]:
        cap = None
        out = None
        positions: List[Dict] = []
        current_positions = {}
        start_times = {}
        processed_frames = 0
        successful_detections = 0

        try:
            # Initialize video capture and writer
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_skip = max(1, int(fps * self.frame_interval))

            logger.info(f"Processing video: {frame_width}x{frame_height} @ {fps}fps")
            
            processed_video_path = os.path.join(output_path, f"{job_id}_processed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, 
                                (frame_width, frame_height))

            for frame_number in range(0, frame_count, frame_skip):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    processed_frames += 1
                    timestamp = timedelta(seconds=frame_number / fps)

                    # Process frame
                    keypoint_frame, keypoints, _, bounding_boxes = self.predictor.on_image(
                        frame, f"{output_path}/frame_{frame_number}")
                    
                    if keypoints and len(keypoints) > 0:
                        successful_detections += 1
                        
                        # Format bounding boxes properly
                        formatted_boxes = self._format_bounding_boxes(bounding_boxes) if bounding_boxes else None
                        
                        # Update tracking with both keypoints and bounding boxes
                        updated_keypoints = self.tracker.update(frame, keypoints, formatted_boxes)
                        
                        # Process detections for each person
                        frame_positions = {}
                        for player_id, (keypoint, bbox) in enumerate(zip(updated_keypoints, formatted_boxes or [])):
                            try:
                                position, confidence = self.tracker.find_position([keypoint])
                                
                                if position:
                                    frame_positions[player_id] = position
                                    
                                    if player_id not in current_positions or position != current_positions[player_id]:
                                        if player_id in current_positions:
                                            positions.append({
                                                'position': current_positions[player_id],
                                                'start_time': start_times[player_id],
                                                'end_time': timestamp,
                                                'player_id': player_id,
                                                'confidence': confidence,
                                                'keypoint_quality': self.tracker.calculate_keypoint_quality(np.array(keypoint)),
                                                'is_smoothed': True,
                                                'bounding_box': bbox if bbox else None,
                                                'keypoints': keypoint
                                            })
                                        current_positions[player_id] = position
                                        start_times[player_id] = timestamp

                            except Exception as e:
                                logger.error(f"Error processing player {player_id}: {str(e)}")
                                continue

                        # Visualize frame with all detections
                        processed_frame = self.visualize_detections(
                            frame, 
                            updated_keypoints,
                            formatted_boxes if formatted_boxes else bounding_boxes,
                            frame_positions
                        )
                        out.write(processed_frame)
                        
                    else:
                        out.write(frame)

                    progress_callback(frame_number / frame_count * 100)

                except Exception as e:
                    logger.error(f"Error processing frame {frame_number}: {str(e)}", exc_info=True)
                    if frame is not None:
                        out.write(frame)
                    continue

            # Add final positions
            final_time = timedelta(seconds=frame_count / fps)
            for player_id, position in current_positions.items():
                try:
                    positions.append({
                        'position': position,
                        'start_time': start_times[player_id],
                        'end_time': final_time,
                        'player_id': player_id,
                        'confidence': confidence,
                        'keypoint_quality': self.tracker.calculate_keypoint_quality(
                            np.array(updated_keypoints[player_id])) if player_id < len(updated_keypoints) else 0.0,
                        'is_smoothed': True,
                        'bounding_box': bounding_boxes[player_id][0] if player_id < len(bounding_boxes) else None,
                        'keypoints': updated_keypoints[player_id] if player_id < len(updated_keypoints) else None
                    })
                except Exception as e:
                    logger.error(f"Error adding final position for player {player_id}: {str(e)}")
                    continue

            logger.info(f"Processed {processed_frames}/{frame_count} frames")
            logger.info(f"Successful detections: {successful_detections}")
            logger.info(f"Total positions detected: {len(positions)}")
            
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
        """Calculate the overall quality of keypoint detections"""
        try:
            if isinstance(keypoints, list):
                keypoints = np.array(keypoints)
            
            keypoints = keypoints.reshape(-1, 3)
            confidences = keypoints[:, 2]
            
            # Calculate various quality metrics
            valid_keypoints = confidences > Config.KEYPOINT_VIZ_THRESHOLD
            num_valid = np.sum(valid_keypoints)
            
            if num_valid == 0:
                return 0.0
            
            # Average confidence of valid keypoints
            avg_confidence = np.mean(confidences[valid_keypoints])
            
            # Coverage (ratio of valid keypoints)
            coverage = num_valid / len(keypoints)
            
            # Spatial distribution of valid keypoints
            if num_valid > 2:
                valid_positions = keypoints[valid_keypoints, :2]
                hull = cv2.convexHull(valid_positions.astype(np.float32))
                hull_area = cv2.contourArea(hull)
                bbox_area = np.ptp(valid_positions[:, 0]) * np.ptp(valid_positions[:, 1])
                spatial_score = min(hull_area / (bbox_area + 1e-6), 1.0)
            else:
                spatial_score = 0.0
            
            # Combine metrics
            quality = (avg_confidence * 0.4 + coverage * 0.4 + spatial_score * 0.2)
            return float(quality)

        except Exception as e:
            logger.error(f"Error calculating keypoint quality: {str(e)}")
            return 0.0


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