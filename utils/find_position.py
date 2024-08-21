import numpy as np
import joblib
import os

def find_position(all_pred_keypoints):
    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), '../trained_model.joblib')
    
    # Load the trained model
    trained_model = joblib.load(model_path)

    # Extract keypoints for player1 and player2
    if len(all_pred_keypoints) >= 2:
      detected_keypoints_player1 = np.array(all_pred_keypoints[0]).flatten()
      detected_keypoints_player2 = np.array(all_pred_keypoints[1]).flatten()

      # Assuming you have a variable 'max_keypoints' defined
      max_keypoints = 18  # Replace with your actual value

      # Take the first max_keypoints from each player's keypoints
      new_data_player1 = detected_keypoints_player1[:max_keypoints * 3]
      new_data_player2 = detected_keypoints_player2[:max_keypoints * 3]

      # Pad with zeros if needed
      new_data_player1 = np.pad(new_data_player1, (0, max_keypoints * 3 - len(new_data_player1)))
      new_data_player2 = np.pad(new_data_player2, (0, max_keypoints * 3 - len(new_data_player2)))

      # Combine the data for prediction
      new_data_combined = np.concatenate((new_data_player1, new_data_player2)).reshape(1, -1)

      # Standardize the new data using mean and standard deviation from training data
      # You need to replace 'mean' and 'std' with the actual mean and standard deviation used during training
      mean = 0.5  # Replace with the mean used during training
      std = 0.2   # Replace with the standard deviation used during training
      new_data_scaled = (new_data_combined - mean) / std

      # Predict the position using the trained model
      predicted_position = trained_model.predict(new_data_scaled)[0]

      return predicted_position
    else:
        # Handle the case when there are fewer than two players detected
        print("Less than two players detected in the frame.")
        return None