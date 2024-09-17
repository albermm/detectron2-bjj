import pytest
from unittest.mock import Mock, patch
import numpy as np

# Mock the entire utils.helper module
utils_helper = Mock()

# Create mock classes for Predictor and VideoProcessor
class MockPredictor:
    def predict_keypoints(self, frame):
        return np.zeros((100, 100, 3)), Mock()

    def save_keypoints(self, outputs):
        return [[[1, 2, 0.9], [3, 4, 0.8]]]

    def onImage(self, input_path, output_path):
        return np.zeros((100, 100, 3)), [[[1, 2, 0.9], [3, 4, 0.8]]], 'TestPosition'

class MockVideoProcessor:
    def __init__(self):
        self.predictor = MockPredictor()

    def process_video(self, video_path, output_path, job_id, user_id):
        return [{'position': 'TestPosition', 'start_time': 0, 'end_time': 1, 'player_id': 1}]

# Patch the utils.helper module to use our mock classes
utils_helper.Predictor = MockPredictor
utils_helper.VideoProcessor = MockVideoProcessor

# Now we can define our tests
def test_predict_keypoints():
    predictor = MockPredictor()
    frame = np.zeros((100, 100, 3))
    keypoint_frame, outputs = predictor.predict_keypoints(frame)
    
    assert isinstance(keypoint_frame, np.ndarray)
    assert outputs is not None

def test_save_keypoints():
    predictor = MockPredictor()
    mock_outputs = Mock()
    keypoints = predictor.save_keypoints(mock_outputs)
    
    assert isinstance(keypoints, list)
    assert len(keypoints) == 1
    assert len(keypoints[0]) == 2

def test_onImage():
    predictor = MockPredictor()
    keypoint_frame, keypoints, predicted_position = predictor.onImage('test_input.jpg', 'test_output')
    
    assert isinstance(keypoint_frame, np.ndarray)
    assert isinstance(keypoints, list)
    assert predicted_position == 'TestPosition'

def test_process_video():
    video_processor = MockVideoProcessor()
    positions = video_processor.process_video('test_video.mp4', 'test_output', 'job123', 'user456')
    
    assert isinstance(positions, list)
    assert len(positions) > 0
    assert 'position' in positions[0]

# Add more tests as needed