import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# Open CSV file to store hand landmark data
with open("hand_landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label"] + [f"x{i},y{i}" for i in range(21)])  # Header row

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)  # X-coordinate
                    landmarks.append(lm.y)  # Y-coordinate

                # Ask for the label (A-Z)
                label = input("Enter the letter (A-Z) for this sign: ").upper()
                writer.writerow([label] + landmarks)  # Save data

        cv2.imshow("Hand Landmark Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
