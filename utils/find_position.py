import numpy as np
import joblib
import os
from typing import List, Optional, Tuple
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
        self.recent_predictions = deque(maxlen=5)  # Store last 5 predictions
        self.last_keypoints = None
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

def find_position(all_pred_keypoints: List[List[float]]) -> Tuple[Optional[str], float]:
    predictor = PositionPredictor()
    return predictor.find_position(all_pred_keypoints)