import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from .shared_utils import logger, Config

class PositionPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PositionPredictor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.trained_model = None
        self.last_keypoints = None
        self.recent_predictions = deque(maxlen=5)  # Store last 5 predictions
        self.reload_model()

    def reload_model(self) -> bool:
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', Config.MODEL_PATH)
            self.trained_model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {Config.MODEL_PATH}: {str(e)}")
            self.trained_model = None
            return False

    def find_position(self, all_pred_keypoints: List[List[float]]) -> Tuple[Optional[str], float]:
        try:
            if len(all_pred_keypoints) < 1:
                logger.warning("No players detected in the frame.")
                return self._smooth_predictions()

            processed_keypoints = []
            keypoint_qualities = []
            for keypoints in all_pred_keypoints[:2]:  # Process up to 2 players
                detected_keypoints = np.array(keypoints).flatten()
                padded_keypoints = np.pad(detected_keypoints[:Config.MAX_KEYPOINTS * 3], 
                                          (0, Config.MAX_KEYPOINTS * 3 - len(detected_keypoints[:Config.MAX_KEYPOINTS * 3])))
                processed_keypoints.append(padded_keypoints)
                keypoint_qualities.append(self._check_keypoint_quality(padded_keypoints))

            if len(processed_keypoints) < 2:
                processed_keypoints.append(np.zeros(Config.MAX_KEYPOINTS * 3))
                keypoint_qualities.append(0.0)

            new_data_combined = np.concatenate(processed_keypoints).reshape(1, -1)

            if np.array_equal(new_data_combined, self.last_keypoints):
                return self._smooth_predictions()

            new_data_scaled = (new_data_combined - Config.KEYPOINT_MEAN) / Config.KEYPOINT_STD

            if self.trained_model is None:
                logger.error("Model not loaded. Attempting to reload.")
                if not self.reload_model():
                    return self._smooth_predictions()

            predicted_position = self.trained_model.predict(new_data_scaled)[0]
            confidence_score = self._calculate_confidence(new_data_scaled)
            
            # Adjust confidence based on keypoint quality
            avg_keypoint_quality = np.mean(keypoint_qualities)
            adjusted_confidence = confidence_score * avg_keypoint_quality

            self.recent_predictions.append((predicted_position, adjusted_confidence))
            self.last_keypoints = new_data_combined

            return self._smooth_predictions()

        except Exception as e:
            logger.error(f"Error in find_position: {str(e)}")
            return self._smooth_predictions()

    def _calculate_confidence(self, scaled_data: np.ndarray) -> float:
        try:
            probabilities = self.trained_model.predict_proba(scaled_data)[0]
            max_prob = np.max(probabilities)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            return max_prob * (1 - entropy / np.log(len(probabilities)))
        except:
            logger.warning("Failed to calculate confidence score. Using default.")
            return 0.5

    def _check_keypoint_quality(self, keypoints: np.ndarray) -> float:
        non_zero = np.count_nonzero(keypoints)
        return non_zero / len(keypoints)

    def _smooth_predictions(self) -> Tuple[Optional[str], float]:
        if not self.recent_predictions:
            return None, 0.0
        positions, confidences = zip(*self.recent_predictions)
        avg_position = max(set(positions), key=positions.count)  # Most common position
        avg_confidence = sum(confidences) / len(confidences)
        return avg_position, avg_confidence

class CombinedTracker:
    def __init__(self, max_frames_to_keep=30):
        self.trackers: Dict[int, cv2.Tracker] = {}
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        self.frame_counts: Dict[int, int] = {}
        self.max_frames_to_keep = max_frames_to_keep
        self.last_positions: Dict[int, np.ndarray] = {}
        self.position_predictor = PositionPredictor()

    def update(self, frame: np.ndarray, keypoints: List[List[float]]) -> List[List[float]]:
        updated_keypoints = []
        for player_id, keypoint in enumerate(keypoints):
            try:
                if player_id not in self.trackers:
                    self._initialize_tracker(player_id, frame, keypoint)
                
                if player_id in self.trackers and player_id in self.kalman_filters:
                    success, box = self.trackers[player_id].update(frame)
                    if success:
                        center_x = box[0] + box[2] / 2
                        center_y = box[1] + box[3] / 2
                        
                        # Kalman filter prediction and update
                        self.kalman_filters[player_id].predict()
                        self.kalman_filters[player_id].update(np.array([[center_x], [center_y]]))
                        
                        estimated_state = self.kalman_filters[player_id].x
                        updated_keypoint = self.update_keypoint_with_estimate(keypoint, estimated_state)
                    else:
                        logger.warning(f"Tracking failed for player {player_id}, using original keypoint")
                        updated_keypoint = keypoint
                else:
                    logger.warning(f"Tracker or Kalman filter not initialized for player {player_id}")
                    updated_keypoint = keypoint
                
                updated_keypoints.append(updated_keypoint)
            except Exception as e:
                logger.error(f"Error updating tracker for player {player_id}: {str(e)}")
                updated_keypoints.append(keypoint)  # Use original keypoint in case of error
        
        return updated_keypoints

    def _initialize_tracker(self, player_id: int, frame: np.ndarray, keypoint: List[float]):
        try:
            self.trackers[player_id] = cv2.TrackerCSRT_create()
            box = self.keypoint_to_box(keypoint)
            if box is None:
                logger.warning(f"Failed to initialize tracker for player {player_id} due to invalid keypoints")
                return

            success = self.trackers[player_id].init(frame, box)
            if not success:
                logger.warning(f"Failed to initialize OpenCV tracker for player {player_id}")
                return

            # Initialize Kalman filter
            kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
            first_x, first_y = self.get_first_valid_coordinates(keypoint)
            kf.x = np.array([first_x, first_y, 0, 0]).reshape((4, 1))
            kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            kf.P *= 1000
            kf.R *= 0.1
            kf.Q *= 0.1
            self.kalman_filters[player_id] = kf
            logger.info(f"Successfully initialized tracker and Kalman filter for player {player_id}")
        except Exception as e:
            logger.error(f"Error initializing tracker for player {player_id}: {str(e)}")

    def _kalman_update(self, player_id: int, measurement: np.ndarray) -> np.ndarray:
        kf = self.kalman_filters[player_id]
        kf.predict()
        kf.update(measurement[:2])
        return kf.x[:2].flatten()

    def update_keypoint_with_estimate(self, keypoint: List[float], estimated_state: np.ndarray) -> List[float]:
        updated_keypoint = keypoint.copy()
        keypoint_array = np.array(keypoint).reshape(-1, 3)
        valid_indices = keypoint_array[:, 2] > 0
        
        if np.any(valid_indices):
            # Update only valid keypoints
            keypoint_array[valid_indices, 0] = estimated_state[0, 0]
            keypoint_array[valid_indices, 1] = estimated_state[1, 0]
        
        return keypoint_array.flatten().tolist()

    def keypoint_to_box(self, keypoint: List[float]) -> Optional[Tuple[int, int, int, int]]:
        if not keypoint or len(keypoint) == 0:
            logger.warning("Empty keypoint received in keypoint_to_box")
            return None
        
        keypoint_array = np.array(keypoint).reshape(-1, 3)  # Reshape to [n, 3] where each row is [x, y, confidence]
        valid_keypoints = keypoint_array[keypoint_array[:, 2] > 0]  # Filter keypoints with confidence > 0
        
        if len(valid_keypoints) == 0:
            logger.warning("No valid keypoints found in keypoint_to_box")
            return None
        
        x_min, y_min = np.min(valid_keypoints[:, :2], axis=0)
        x_max, y_max = np.max(valid_keypoints[:, :2], axis=0)
        
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

    def adjust_keypoints(self, keypoint: List[float], box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        if w == 0 or h == 0:
            logger.warning("Box with zero width or height in adjust_keypoints")
            return np.array(keypoint)
        keypoint_array = np.array(keypoint)
        scaled_keypoint = (keypoint_array - [x, y]) / [w, h]
        adjusted_keypoint = scaled_keypoint * [w, h] + [x, y]
        return adjusted_keypoint

    def get_confidence(self, player_id: int) -> float:
        return max(0, 1 - (self.frame_counts.get(player_id, 0) / self.max_frames_to_keep))

    def predict_position(self, player_id: int) -> np.ndarray:
        if player_id in self.kalman_filters:
            kf = self.kalman_filters[player_id]
            kf.predict()
            return kf.x[:2].flatten()
        return self.last_positions.get(player_id, None)

    def reset_tracker(self, player_id: int, frame: np.ndarray, keypoint: List[float]):
        if player_id in self.trackers:
            del self.trackers[player_id]
        self._initialize_tracker(player_id, frame, keypoint)
        self.frame_counts[player_id] = 0
        self.last_positions[player_id] = np.array(keypoint)
        logger.info(f"Reset tracker for player {player_id}")

    def _remove_tracker(self, player_id: int):
        logger.info(f"Removing tracker for player {player_id}")
        del self.trackers[player_id]
        del self.frame_counts[player_id]
        del self.last_positions[player_id]
        if player_id in self.kalman_filters:
            del self.kalman_filters[player_id]

    def find_position(self, all_pred_keypoints: List[List[float]]) -> Tuple[Optional[str], float]:
        return self.position_predictor.find_position(all_pred_keypoints)