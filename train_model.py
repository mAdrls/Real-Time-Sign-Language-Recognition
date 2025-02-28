import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv("hand_landmarks_dataset.csv")
X = df.drop(columns=["Label"])
y = df["Label"]

# Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the trained model
joblib.dump(knn, "hand_sign_model.pkl")
print("Model trained and saved as hand_sign_model.pkl")
