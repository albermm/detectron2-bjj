import pytest
from utils.helper import Predictor
from utils.find_position import find_position

def test_predictor_and_find_position_integration():
    predictor = Predictor()
    test_image_path = 'path/to/test/image.jpg'
    output_path = 'path/to/test/output'
    
    keypoint_frame, keypoints, _ = predictor.onImage(test_image_path, output_path)
    
    assert keypoints is not None, "Keypoints should not be None"
    
    predicted_position = find_position(keypoints)
    assert predicted_position is not None, "Predicted position should not be None"
    assert isinstance(predicted_position, str), "Predicted position should be a string"