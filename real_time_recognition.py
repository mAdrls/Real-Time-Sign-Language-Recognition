import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained mod
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


