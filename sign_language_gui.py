# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import tkinter as tk
# from tkinter import scrolledtext
# import time
# from collections import deque

# # Load trained model
# model = joblib.load("hand_sign_model.pkl")

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Initialize GUI
# root = tk.Tk()
# root.title("Sign Language Recognition")

# # Text Display Box
# text_display = scrolledtext.ScrolledText(root, width=40, height=5, font=("Arial", 14))
# text_display.pack()

# # Functions for Buttons
# def clear_text():
#     text_display.delete(1.0, tk.END)

# def save_text():
#     with open("recognized_text.txt", "w") as file:
#         file.write(text_display.get(1.0, tk.END))

# def quit_app():
#     root.destroy()
#     cap.release()
#     cv2.destroyAllWindows()

# # Buttons
# btn_frame = tk.Frame(root)
# btn_frame.pack()

# clear_btn = tk.Button(btn_frame, text="Clear All", command=clear_text, bg="lightgray")
# clear_btn.pack(side=tk.LEFT, padx=5)

# save_btn = tk.Button(btn_frame, text="Save to a Text File", command=save_text, bg="green", fg="white")
# save_btn.pack(side=tk.LEFT, padx=5)

# quit_btn = tk.Button(btn_frame, text="Quit", command=quit_app, bg="red", fg="white")
# quit_btn.pack(side=tk.LEFT, padx=5)

# # Start webcam
# cap = cv2.VideoCapture(0)

# last_prediction = ""
# last_time = time.time()
# prediction_queue = deque(maxlen=5)  # Store last 5 predictions

# def recognize_sign():
#     global last_prediction, last_time, prediction_queue

#     ret, frame = cap.read()
#     if not ret:
#         return

#     # Convert to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             # Extract hand landmarks
#             hand_data = []
#             for landmark in hand_landmarks.landmark:
#                 hand_data.extend([landmark.x, landmark.y])

#             # Predict using trained model
#             hand_data = np.array(hand_data).reshape(1, -1)
#             prediction = model.predict(hand_data)[0]
#             prediction_queue.append(prediction)

#             # Apply smoothing (only accept stable predictions)
#             if len(set(prediction_queue)) == 1:  # If last 5 predictions are the same
#                 current_time = time.time()
#                 if prediction != last_prediction:
#                     if current_time - last_time > 1:  # Add space if 1 second passed
#                         text_display.insert(tk.END, " ")
#                     text_display.insert(tk.END, prediction)
#                     last_prediction = prediction
#                     last_time = current_time

#             # Draw hand landmarks
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # Show webcam output
#     cv2.imshow("Hand Sign Recognition", frame)
#     root.after(10, recognize_sign)

# # Start recognition loop
# root.after(10, recognize_sign)
# root.mainloop()
