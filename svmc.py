import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(64, 64)):
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if the image is loaded successfully
        if image is None:
            raise FileNotFoundError(f"Failed to load image at path: {image_path}")
        
        # Resize, convert to grayscale, and normalize
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        
        return image
    except Exception as e:
        print(f"Error processing image at path {image_path}: {e}")
        return None

# Example usage
directory_path = r"C:\Users\temar\Downloads\archive (2)\valid"
image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]  # Filter JPG files
images = []

for image_file in image_files:
    image_path = os.path.join(directory_path, image_file)
    preprocessed_image = preprocess_image(image_path)
    
    if preprocessed_image is not None:
        images.append(preprocessed_image)

images_array = np.array(images)

# Print the shape of the images array to verify
print(images_array.shape)

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load and preprocess images (similar to your existing code)
def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image.astype('float32') / 255.0  # Normalize
    return image

directory_path = r"C:\Users\temar\Downloads\archive (2)\valid"
image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
images = []

for image_file in image_files:
    image_path = os.path.join(directory_path, image_file)
    preprocessed_image = preprocess_image(image_path)
    
    if preprocessed_image is not None:
        images.append(preprocessed_image)

images_array = np.array(images)

# Load or create labels for the images (dummy example)
labels = np.random.randint(0, 2, size=len(images))  # Dummy labels (0 or 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_array, labels, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train SVM model
svm_classifier.fit(X_train.reshape(len(X_train), -1), y_train)

# Make predictions
y_pred_svm = svm_classifier.predict(X_test.reshape(len(X_test), -1))

# Evaluate performance
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)
print("SVM Classification Report:\n", report_svm)

from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression classifier
logreg_classifier = LogisticRegression(random_state=42)

# Train Logistic Regression model
logreg_classifier.fit(X_train.reshape(len(X_train), -1), y_train)

# Make predictions
y_pred_logreg = logreg_classifier.predict(X_test.reshape(len(X_test), -1))

# Evaluate performance
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
report_logreg = classification_report(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", accuracy_logreg)
print("Logistic Regression Classification Report:\n", report_logreg)

