import numpy as np
from typing import Dict, Optional, Union
from .shared_utils import logger

def assign_players(
    predicted_positions_player1: Union[np.ndarray, float, int],
    predicted_positions_player2: Union[np.ndarray, float, int],
    current_position_player1: Optional[np.ndarray],
    current_position_player2: Optional[np.ndarray],
    position_change_threshold: float = 0.1
) -> Dict[int, np.ndarray]:
    try:
        predicted_positions_player1 = np.atleast_1d(predicted_positions_player1)
        predicted_positions_player2 = np.atleast_1d(predicted_positions_player2)
        current_position_player1 = np.atleast_1d(current_position_player1) if current_position_player1 is not None else None
        current_position_player2 = np.atleast_1d(current_position_player2) if current_position_player2 is not None else None
    except ValueError as e:
        logger.error(f"Error converting inputs to numpy arrays: {str(e)}")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}

    if any(pos is None or pos.size == 0 for pos in [predicted_positions_player1, predicted_positions_player2]):
        logger.warning("Invalid input: predicted positions are None or empty")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}

    if predicted_positions_player1.shape != predicted_positions_player2.shape:
        logger.warning(f"Predicted positions have different shapes: {predicted_positions_player1.shape} vs {predicted_positions_player2.shape}")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}

    try:
        if current_position_player1 is None or current_position_player2 is None:
            logger.info("No current positions available, returning predicted positions")
            return {1: predicted_positions_player1, 2: predicted_positions_player2}

        if current_position_player1.shape != current_position_player2.shape:
            logger.warning(f"Current positions have different shapes: {current_position_player1.shape} vs {current_position_player2.shape}")
            return {1: predicted_positions_player1, 2: predicted_positions_player2}

        dist_player1 = np.linalg.norm(predicted_positions_player1 - current_position_player1)
        dist_player2 = np.linalg.norm(predicted_positions_player2 - current_position_player2)

        if abs(dist_player1 - dist_player2) < position_change_threshold:
            logger.info("Position change below threshold, maintaining current assignments")
            return {1: predicted_positions_player1, 2: predicted_positions_player2}
        elif dist_player1 < dist_player2:
            return {1: predicted_positions_player1, 2: predicted_positions_player2}
        else:
            return {1: predicted_positions_player2, 2: predicted_positions_player1}
    except np.linalg.LinAlgError as e:
        logger.error(f"Linear algebra error in distance calculation: {str(e)}")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}
    except Exception as e:
        logger.error(f"Unexpected error in assign_players: {str(e)}")
        return {1: predicted_positions_player1, 2: predicted_positions_player2}