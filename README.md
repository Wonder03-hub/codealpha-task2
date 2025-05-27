#TASK 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load MNIST dataset from sklearn
digits = datasets.load_digits()

# X - Features (images), y - Labels (digit labels)
X = digits.data
y = digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (important for SVMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
clf = SVC(kernel='linear')

# Train the model
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot an image from the test set and the predicted label
plt.figure(figsize=(6, 3))

# Plot the first test image
plt.subplot(1, 2, 1)
plt.imshow(X_test[0].reshape(8, 8), cmap='gray')
plt.title(f"True Label: {y_test[0]}\nPredicted: {y_pred[0]}")
plt.axis('off')

# Plot the second test image
plt.subplot(1, 2, 2)
plt.imshow(X_test[1].reshape(8, 8), cmap='gray')
plt.title(f"True Label: {y_test[1]}\nPredicted: {y_pred[1]}")
plt.axis('off')

plt.show()

# Plot the first 5 test images and their predictions
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
