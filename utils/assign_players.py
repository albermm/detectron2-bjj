import numpy as np
from typing import Dict, Optional
from .shared_utils import logger

def assign_players(
    predicted_positions_player1: np.ndarray,
    predicted_positions_player2: np.ndarray,
    current_position_player1: Optional[np.ndarray],
    current_position_player2: Optional[np.ndarray]
) -> Dict[int, np.ndarray]:
    # Convert inputs to numpy arrays if they're not already
    try:
        predicted_positions_player1 = np.array(predicted_positions_player1)
        predicted_positions_player2 = np.array(predicted_positions_player2)
        current_position_player1 = np.array(current_position_player1) if current_position_player1 is not None else None
        current_position_player2 = np.array(current_position_player2) if current_position_player2 is not None else None
    except Exception as e:
        logger.error(f"Error converting inputs to numpy arrays in assign_players: {str(e)}")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}

    # Check if any of the inputs are None or empty
    if any(pos is None or len(pos) == 0 for pos in [predicted_positions_player1, predicted_positions_player2]):
        logger.warning("Invalid input types in assign_players: predicted positions are None or empty")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}

    # Check if predicted positions have different shapes
    if predicted_positions_player1.shape != predicted_positions_player2.shape:
        logger.warning("Predicted positions have different shapes")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}

    try:
        # If we don't have current positions, just return the predicted positions
        if current_position_player1 is None or current_position_player2 is None:
            return {1: predicted_positions_player1, 2: predicted_positions_player2}

        # Calculate Euclidean distance between predicted and current poses
        dist_player1 = np.linalg.norm(predicted_positions_player1 - current_position_player1)
        dist_player2 = np.linalg.norm(predicted_positions_player2 - current_position_player2)

        # Assign poses based on minimizing distance
        if dist_player1 < dist_player2:
            return {1: predicted_positions_player1, 2: predicted_positions_player2}
        else:
            return {1: predicted_positions_player2, 2: predicted_positions_player1}
    except Exception as e:
        logger.error(f"Error in assign_players: {str(e)}")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}