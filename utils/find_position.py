import numpy as np
import joblib
import os
from shared_utils import logger, Config

def find_position(all_pred_keypoints):
    try:
        # Load the trained model
        model_path = os.path.join(os.path.dirname(__file__), Config.MODEL_PATH)
        
        # Load the trained model
        trained_model = joblib.load(model_path)

        # Extract keypoints for player1 and player2
        if len(all_pred_keypoints) >= 2:
            detected_keypoints_player1 = np.array(all_pred_keypoints[0]).flatten()
            detected_keypoints_player2 = np.array(all_pred_keypoints[1]).flatten()

            # Take the first max_keypoints from each player's keypoints
            new_data_player1 = detected_keypoints_player1[:Config.MAX_KEYPOINTS * 3]
            new_data_player2 = detected_keypoints_player2[:Config.MAX_KEYPOINTS * 3]

            # Pad with zeros if needed
            new_data_player1 = np.pad(new_data_player1, (0, Config.MAX_KEYPOINTS * 3 - len(new_data_player1)))
            new_data_player2 = np.pad(new_data_player2, (0, Config.MAX_KEYPOINTS * 3 - len(new_data_player2)))

            # Combine the data for prediction
            new_data_combined = np.concatenate((new_data_player1, new_data_player2)).reshape(1, -1)

            # Standardize the new data using mean and standard deviation from training data
            new_data_scaled = (new_data_combined - Config.KEYPOINT_MEAN) / Config.KEYPOINT_STD

            # Predict the position using the trained model
            predicted_position = trained_model.predict(new_data_scaled)[0]

            return predicted_position
        else:
            logger.warning("Less than two players detected in the frame.")
            return None
    except Exception as e:
        logger.error(f"Error in find_position: {str(e)}")
        raise

# TODO: Implement unit tests for find_position function