import cv2
import mediapipe as mp
import os
# import numpy as np
import pandas as pd

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Folder containing ASL images
DATASET_PATH = "dataset"  # Update this to your dataset folder
labels = []
landmarks = []

# Process each image
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmark coordinates
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y])
                landmarks.append(hand_data)
                labels.append(label)

# Convert to CSV file
df = pd.DataFrame(landmarks)
df["Label"] = labels
df.to_csv("hand_landmarks_dataset.csv", index=False)
print("Dataset saved as hand_landmarks_dataset.csv")
