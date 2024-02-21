import numpy as np



def predict_positions(model, scaler, new_image_pose1, new_image_pose2):
    # Combine Pose1 and Pose2 for the new image into a single input
    new_image_combined = np.concatenate((new_image_pose1, new_image_pose2), axis=1)

    # Scale the features using the same scaler used during training
    new_image_scaled = scaler.transform(new_image_combined)

    # Use the trained model to predict positions
    predicted_positions = model.predict(new_image_scaled)
    print("Predicted Positions:", predicted_positions)
    return predicted_positions
