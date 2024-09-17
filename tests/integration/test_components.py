import cv2
import numpy as np
from utils.helper import Predictor
from utils.find_position import find_position
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_predictor_and_find_position_integration():
    predictor = Predictor()
    test_image_path = '/Users/albertomartin/Coding/detectron_bjj/data/inputs/0000168.jpg'
    output_path = 'data/output'

    # Check if the image file exists
    assert os.path.exists(test_image_path), f"Image file does not exist: {test_image_path}"

    # Debug: Check if the image can be read
    image = cv2.imread(test_image_path)
    assert image is not None, f"Failed to load image from {test_image_path}"
    logger.debug(f"Image shape: {image.shape}")

    try:
        keypoint_frame, keypoints, predicted_position = predictor.onImage(test_image_path, output_path)
    except Exception as e:
        logger.error(f"Error in predictor.onImage: {str(e)}")
        raise

    # Debug: Print keypoint_frame type and shape
    logger.debug(f"keypoint_frame type: {type(keypoint_frame)}")
    if isinstance(keypoint_frame, np.ndarray):
        logger.debug(f"keypoint_frame shape: {keypoint_frame.shape}")
    else:
        logger.warning(f"keypoint_frame is not a numpy array: {keypoint_frame}")

    logger.debug(f"keypoints: {keypoints}")
    logger.debug(f"predicted_position: {predicted_position}")

    assert keypoints is not None, "Keypoints should not be None"
    
    # Check if enough players were detected for position prediction
    if len(keypoints) >= 2:
        assert predicted_position is not None, "Predicted position should not be None"
        assert isinstance(predicted_position, str), "Predicted position should be a string"
    else:
        logger.warning("Not enough players detected for position prediction.")
        assert predicted_position is None, "Predicted position should be None when not enough players are detected"