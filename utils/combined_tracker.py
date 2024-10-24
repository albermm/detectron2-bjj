import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Dict, Optional
from .shared_utils import logger, Config
from .find_position import PositionPredictor

class CombinedTracker:
    def __init__(self):
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self.position_predictor = PositionPredictor()

    def update(self, frame: np.ndarray, keypoints: List[List[float]], bounding_boxes: List[List[float]] = None) -> List[List[float]]:
        """
        Update tracking with new frame data
        Args:
            frame: Current video frame
            keypoints: List of keypoint sets for each detected person
            bounding_boxes: Optional list of bounding boxes for each person
        """
        updated_keypoints = []
        
        if bounding_boxes is None:
            # If no bounding boxes provided, create from keypoints
            bounding_boxes = [self.keypoint_to_box(kps) for kps in keypoints]
        
        for player_id, (keypoint, bbox) in enumerate(zip(keypoints, bounding_boxes)):
            try:
                if player_id not in self.trackers or self.should_reinitialize(player_id):
                    success = self._initialize_tracker(player_id, frame, keypoint)
                    if not success:
                        updated_keypoints.append(keypoint)
                        continue

                # Update tracking
                if player_id in self.trackers:
                    success, box = self.trackers[player_id].update(frame)
                else:
                    success = False

                if success:
                    # Use Kalman filter
                    center_x = box[0] + box[2] / 2
                    center_y = box[1] + box[3] / 2
                    self.kalman_filters[player_id].predict()
                    self.kalman_filters[player_id].update(np.array([[center_x], [center_y]]))
                    estimated_state = self.kalman_filters[player_id].x
                    updated_keypoint = self.update_keypoint_with_estimate(keypoint, estimated_state)
                else:
                    updated_keypoint = self.update_fallback_tracker(player_id, keypoint)

                updated_keypoints.append(updated_keypoint)
                self.last_reinitialization[player_id] += 1

            except Exception as e:
                logger.error(f"Error updating tracker for player {player_id}: {str(e)}")
                updated_keypoints.append(keypoint)

        return updated_keypoints

    def _initialize_kalman_filter(self, player_id: int, keypoint: List[float], box: List[float]):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        kf.x = np.array([center_x, center_y, 0, 0]).reshape((4, 1))
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R *= 0.1
        kf.Q *= 0.1

        self.kalman_filters[player_id] = kf

    def _update_keypoints(self, keypoint: List[float], estimate: np.ndarray) -> List[float]:
        updated_keypoint = np.array(keypoint).reshape(-1, 3)
        center = np.mean(updated_keypoint[:, :2], axis=0)
        offset = estimate - center
        updated_keypoint[:, :2] += offset
        return updated_keypoint.flatten().tolist()

    def _update_bounding_box(self, box: List[float], estimate: np.ndarray) -> List[float]:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        new_center_x, new_center_y = estimate
        new_x1 = new_center_x - width / 2
        new_y1 = new_center_y - height / 2
        new_x2 = new_center_x + width / 2
        new_y2 = new_center_y + height / 2
        
        return [new_x1, new_y1, new_x2, new_y2]

    def analyze_trajectory(self, player_id: int) -> Optional[str]:
        if player_id not in self.trajectories or len(self.trajectories[player_id]) < 2:
            return None

        trajectory = np.array(self.trajectories[player_id])
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

        if displacement < 10:  # Threshold for stationary
            return "stationary"
        elif total_distance > displacement * 1.5:  # Threshold for erratic
            return "erratic"
        else:
            return "linear"

    def find_position(self, keypoints: List[List[float]]) -> Tuple[Optional[str], float]:
        return self.position_predictor.find_position(keypoints)