import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import Label, Button, filedialog, Frame

# Load trained model
model = joblib.load("hand_sign_model.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Create Tkinter UI
root = tk.Tk()
root.title("Hand Sign Recognition")
root.geometry("600x400")

recognized_text = ""

# Function to update text display
def update_text(prediction):
    global recognized_text
    recognized_text += prediction + " "
    label_text.config(text=recognized_text)

# Function to remove last word or letter
def remove_last():
    global recognized_text
    words = recognized_text.strip().split(" ")
    if words:
        words.pop()
        recognized_text = " ".join(words) + " "
        label_text.config(text=recognized_text)

# Function to clear text
def clear_text():
    global recognized_text
    recognized_text = ""
    label_text.config(text=recognized_text)

# Function to save text to a file
def save_text():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "w") as file:
            file.write(recognized_text)

# Function to exit application
def exit_app():
    root.destroy()
    cap.release()
    cv2.destroyAllWindows()

# UI Layout
frame_buttons = Frame(root)
frame_buttons.pack(pady=10)

label_text = Label(root, text="", font=("Arial", 14))
label_text.pack(pady=20)

capture_button = Button(frame_buttons, text="Capture", command=lambda: update_text(current_prediction), font=("Arial", 12))
capture_button.grid(row=0, column=0, padx=5, pady=5)

remove_button = Button(frame_buttons, text="Remove Last", command=remove_last, font=("Arial", 12))
remove_button.grid(row=0, column=1, padx=5, pady=5)

clear_button = Button(frame_buttons, text="Clear", command=clear_text, font=("Arial", 12))
clear_button.grid(row=0, column=2, padx=5, pady=5)

save_button = Button(frame_buttons, text="Save", command=save_text, font=("Arial", 12))
save_button.grid(row=1, column=0, padx=5, pady=5)

exit_button = Button(frame_buttons, text="Exit", command=exit_app, font=("Arial", 12))
exit_button.grid(row=1, column=1, padx=5, pady=5)

# Webcam loop
current_prediction = ""
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
            current_prediction = model.predict(hand_data)[0]

            # Display prediction on webcam
            cv2.putText(frame, f"Prediction: {current_prediction}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show output
    cv2.imshow("Hand Sign Recognition", frame)
    root.update()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
root.mainloop()
