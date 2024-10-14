import numpy as np

def assign_players(predicted_positions_player1, predicted_positions_player2,
                   annotated_positions_player1, annotated_positions_player2):
    # Check if any of the inputs are None or not numeric
    if any(not isinstance(pos, (np.ndarray, list)) for pos in [predicted_positions_player1, predicted_positions_player2, annotated_positions_player1, annotated_positions_player2]):
        print("Error: Invalid input types in assign_players")
        return predicted_positions_player1, predicted_positions_player2

    try:
        # Convert inputs to numpy arrays if they're not already
        predicted_positions_player1 = np.array(predicted_positions_player1)
        predicted_positions_player2 = np.array(predicted_positions_player2)
        annotated_positions_player1 = np.array(annotated_positions_player1)
        annotated_positions_player2 = np.array(annotated_positions_player2)

        # Calculate Euclidean distance between predicted and annotated poses
        dist_player1 = np.linalg.norm(predicted_positions_player1 - annotated_positions_player1)
        dist_player2 = np.linalg.norm(predicted_positions_player2 - annotated_positions_player2)

        # Assign poses based on minimizing distance
        if dist_player1 < dist_player2:
            return predicted_positions_player1, predicted_positions_player2
        else:
            return predicted_positions_player2, predicted_positions_player1
    except Exception as e:
        print(f"Error in assign_players: {str(e)}")
        return predicted_positions_player1, predicted_positions_player2