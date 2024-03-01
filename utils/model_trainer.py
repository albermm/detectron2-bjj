import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_annotations(json_file):
    with open(json_file, 'r') as file:
        annotations = json.load(file)
    return annotations

json_file_path = '/content/annotations.json'
annotations = load_annotations(json_file_path)

def prepare_data(annotations, max_keypoints=18):
    X = []
    y = []

    for annotation in annotations:
        pose1 = np.array(annotation.get("pose1", [])).flatten()
        pose2 = np.array(annotation.get("pose2", [])).flatten()

        # Take the first max_keypoints from each pose
        pose1 = pose1[:max_keypoints * 3]
        pose2 = pose2[:max_keypoints * 3]

        # Pad with zeros if there are fewer than max_keypoints keypoints
        pose1 = np.pad(pose1, (0, max_keypoints * 3 - len(pose1)))
        pose2 = np.pad(pose2, (0, max_keypoints * 3 - len(pose2)))

        # Combine pose1 and pose2
        combined_pose = np.concatenate((pose1, pose2))

        X.append(combined_pose)
        y.append(annotation["position"])

    X = np.array(X)
    return X, y


def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter as needed
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


    # Save the trained model
    joblib.dump(model, 'trained_model.joblib')
    print("Model saved successfully.")

    return model

# Load annotations from the file
#annotations = load_annotations("/content/annotations.json")

# Prepare data for training
X, y = prepare_data(annotations)

# Train the model
trained_model = train_model(X, y)

