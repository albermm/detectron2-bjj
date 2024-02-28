import numpy as np
from utils.assign_players import assign_players

def predict_positions(model, new_image_pose1, new_image_pose2):
    # Combine Pose1 and Pose2 for the new image into a single input
    new_image_combined = np.concatenate((new_image_pose1, new_image_pose2))

    # Use the trained model to predict positions
    predicted_positions = model.predict([new_image_combined])

    return predicted_positions

def match_keypoints(trained_model, annotations, predictions):
    matches = []

    for frame_prediction in predictions:
        frame_index = frame_prediction['frame_index']
        predicted_positions_player1 = frame_prediction['player1_positions']
        predicted_positions_player2 = frame_prediction['player2_positions']

        # Find the corresponding annotated frame
        annotated_frame = next((ann for ann in annotations if ann['frame_index'] == frame_index), None)

        if annotated_frame:
            annotated_positions_player1 = np.array(annotated_frame.get("pose1", [])).flatten()
            annotated_positions_player2 = np.array(annotated_frame.get("pose2", [])).flatten()

            # Use the assign_players function to maintain consistency
            assigned_positions_player1, assigned_positions_player2 = assign_players(
                predicted_positions_player1, predicted_positions_player2,
                annotated_positions_player1, annotated_positions_player2
            )

            # Compare the predicted positions with annotated positions
            match_player1 = np.array_equal(assigned_positions_player1, annotated_positions_player1)
            match_player2 = np.array_equal(assigned_positions_player2, annotated_positions_player2)

            matches.append({
                'frame_index': frame_index,
                'match_player1': match_player1,
                'match_player2': match_player2
            })

    return matches

# Example usage
# Note: You need to have 'trained_model', 'annotations', and 'result' variables available
# 'trained_model' is the logistic regression model trained on keypoints
# 'annotations' is the annotated dataset
# 'result' is the output of the process_video function

# Call the matching function
matching_result = match_keypoints(trained_model, annotations, result)

# Now 'matching_result' contains a summary of matches for each frame
# Analyze 'matching_result' to evaluate the model's performance
