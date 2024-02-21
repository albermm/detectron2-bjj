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
    
    # Print a sample of annotations
    num_samples_to_print = 5
    print(f"Printing {num_samples_to_print} annotations as a sample:")
    for idx, annotation in enumerate(annotations[:num_samples_to_print]):
        print(f"Annotation {idx + 1}: {annotation}")

    for annotation in annotations:
        # Check for the presence of both "pose1" and "pose2"
        if "pose1" in annotation and "pose2" in annotation:
            # Handle both poses as needed
            pose1 = np.array(annotation["pose1"]).flatten()
            pose2 = np.array(annotation["pose2"]).flatten()
            # Combine or handle them based on your requirements
            combined_pose = np.concatenate((pose1, pose2))
        elif "pose1" in annotation:
            combined_pose = np.array(annotation["pose1"]).flatten()
        elif "pose2" in annotation:
            combined_pose = np.array(annotation["pose2"]).flatten()
        else:
            # Handle the case where neither "pose1" nor "pose2" is present
            continue

        # Assuming 'position' is the label for the position of the athlete
        position = annotation["position"]
        
        # Append the flattened pose data and position label to X and y
        X.append(combined_pose)
        y.append(position)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


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

# Prepare data for training
X, y = prepare_data(annotations)

# Train the model
trained_model = train_model(X, y)
