import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("hand_sign_model.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract hand landmarks
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y])

            # Predict using trained model
            hand_data = np.array(hand_data).reshape(1, -1)
            prediction = model.predict(hand_data)[0]

            # Display prediction
            cv2.putText(frame, f"Prediction: {prediction}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show output
    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# This below code is the 3D GUI Hand Sign Dectetor Code 

# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize MediaPipe Hand module
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# # Set up the 3D plot
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Initial empty scatter plot
# scatter = ax.scatter([], [], [], c='r', marker='o')

# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert BGR to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Extract 3D coordinates
#                 landmarks_3d = []
#                 for lm in hand_landmarks.landmark:
#                     landmarks_3d.append([lm.x, lm.y, lm.z])

#                 landmarks_3d = np.array(landmarks_3d)  # Convert to NumPy array

#                 # Update the 3D plot with new landmarks
#                 scatter._offsets3d = (landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])

#                 # Draw the plot again with updated landmarks
#                 plt.draw()
#                 plt.pause(0.01)

#         cv2.imshow("3D Hand Tracking", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
