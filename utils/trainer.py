import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def prepare_data(annotations):
    X = []
    y = []

    for annotation in annotations:
      

        pose1 = np.array(annotation["Pose1"]).flatten()
        pose2 = np.array(annotation["Pose2"]).flatten()
        combined_pose = np.concatenate((pose1, pose2))
        position = annotation["Position"]

        X.append(combined_pose)
        y.append(position)

    return np.array(X), np.array(y)

def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model (you can use a more sophisticated model)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model

# Load annotations from the file
annotations = load_annotations("/content/annotations.json")
print(f"annotations {annotations}")
# Prepare data for training
X, y = prepare_data(annotations)

# Train the model
trained_model = train_model(X, y)
