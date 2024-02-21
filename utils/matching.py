def predict_positions(model, new_image_pose1, new_image_pose2):
    # Combine Pose1 and Pose2 for the new image into a single input
    new_image_combined = np.concatenate((new_image_pose1, new_image_pose2))

    # Use the trained model to predict positions
    predicted_positions = model.predict([new_image_combined])

    return predicted_positions

# Example: New image keypoints for Pose1 and Pose2
new_image_pose1 = np.array(...)  # Replace with actual keypoints
new_image_pose2 = np.array(...)  # Replace with actual keypoints

# Predict positions for the new image
predicted_positions = predict_positions(trained_model, new_image_pose1, new_image_pose2)

print("Predicted Positions:", predicted_positions)
