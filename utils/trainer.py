import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = json.load(f)
    return annotations

import numpy as np

def prepare_data(annotations, max_keypoints=18):
    X = []
    y = []

    for annotation in annotations:
        pose1 = np.array(annotation.get("pose1", [])).flatten()
        pose2 = np.array(annotation.get("pose2", [])).flatten()

        # Combine pose1 and pose2
        combined_pose = np.concatenate((pose1, pose2))

        # Ensure a fixed size by padding or truncating
        if len(combined_pose) < 2 * max_keypoints:
            combined_pose = np.pad(combined_pose, (0, 2 * max_keypoints - len(combined_pose)))
        elif len(combined_pose) > 2 * max_keypoints:
            combined_pose = combined_pose[:2 * max_keypoints]

        X.append(combined_pose)
        y.append(annotation["position"])

    X = np.array(X)
    return X, y



def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)  # Increase max_iter as needed
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model, scaler

# Load annotations from the file
annotations = load_annotations("/content/annotations.json")

# Prepare data for training
X, y = prepare_data(annotations)

# Train the model
trained_model = train_model(X, y)
