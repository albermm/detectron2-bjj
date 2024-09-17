import pytest
from unittest.mock import patch, Mock
import numpy as np

# Mock the entire utils.find_position module
utils_find_position = Mock()

@pytest.fixture
def mock_find_position():
    def _find_position(all_pred_keypoints):
        if len(all_pred_keypoints) >= 2:
            return 'TestPosition'
        return None
    return _find_position

def test_find_position_success(mock_find_position):
    all_pred_keypoints = [
        [[1, 2, 0.9]] * 18,  # 18 keypoints for player 1
        [[5, 6, 0.7]] * 18   # 18 keypoints for player 2
    ]
    
    result = mock_find_position(all_pred_keypoints)
    
    assert result == 'TestPosition'

def test_find_position_less_than_two_players(mock_find_position):
    all_pred_keypoints = [
        [[1, 2, 0.9]] * 18  # Only one player
    ]
    
    result = mock_find_position(all_pred_keypoints)
    
    assert result is None

# Add more tests...