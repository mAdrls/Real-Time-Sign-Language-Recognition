# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import tkinter as tk
# from tkinter import Label, Button, filedialog, Frame
# import threading

# # Load trained model
# model = joblib.load("hand_sign_model.pkl")

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Start webcam
# cap = cv2.VideoCapture(0)

# # Create Tkinter UI
# root = tk.Tk()
# root.title("Hand Sign Recognition")
# root.geometry("600x400")

# recognized_text = ""  # Stores final recognized text
# recognized_buffer = ""  # Stores recognized letters for suggestions
# current_prediction = ""  # Stores the current hand sign prediction

# common_words = ["hello", "help", "world", "good", "morning", "thank", "you", "please"]

# def suggest_word(buffer):
#     matches = [word for word in common_words if word.startswith(buffer)]
#     return matches[0] if matches else None  # Return first match or None

# # Function to update text display
# def update_text():
#     global recognized_buffer, recognized_text
#     if current_prediction:  # Only update if there's a valid prediction
#         recognized_buffer += current_prediction  # Append the predicted letter to the buffer
#         recognized_text += current_prediction  # Append the predicted letter to the final text
#         label_text.config(text=recognized_text)  # Update the textbox

#         # Update suggestion
#         suggestion = suggest_word(recognized_buffer)
#         if suggestion:
#             suggestion_label.config(text=f"Suggestion: {suggestion}")
#         else:
#             suggestion_label.config(text="")  # Clear suggestion if none

# # Function to accept suggestion
# def accept_suggestion():
#     global recognized_text, recognized_buffer
#     suggestion = suggest_word(recognized_buffer)
#     if suggestion:
#         # Replace the buffer with the suggestion in the final text
#         recognized_text = recognized_text[:-len(recognized_buffer)] + suggestion + " "
#         label_text.config(text=recognized_text)  # Update the textbox
#         recognized_buffer = ""  # Clear the buffer
#         suggestion_label.config(text="")  # Clear the suggestion

# # Function to remove last word or letter
# def remove_last():
#     global recognized_text, recognized_buffer
#     if recognized_text:
#         recognized_text = recognized_text[:-1]  # Remove the last character
#         recognized_buffer = recognized_buffer[:-1]  # Remove the last character from the buffer
#         label_text.config(text=recognized_text)  # Update the textbox

#         # Update suggestion
#         suggestion = suggest_word(recognized_buffer)
#         if suggestion:
#             suggestion_label.config(text=f"Suggestion: {suggestion}")
#         else:
#             suggestion_label.config(text="")  # Clear suggestion if none

# # Function to clear text
# def clear_text():
#     global recognized_text, recognized_buffer
#     recognized_text = ""
#     recognized_buffer = ""
#     label_text.config(text=recognized_text)
#     suggestion_label.config(text="")  # Clear suggestion

# # Function to save text to a file
# def save_text():
#     file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
#     if file_path:
#         with open(file_path, "w") as file:
#             file.write(recognized_text)

# # Function to exit application
# def exit_app():
#     root.quit()  # Stop Tkinter main loop
#     cap.release()  # Release webcam
#     cv2.destroyAllWindows()  # Close OpenCV windows

# # UI Layout
# frame_buttons = Frame(root)
# frame_buttons.pack(pady=10)

# label_text = Label(root, text="", font=("Arial", 14))
# label_text.pack(pady=20)

# suggestion_label = Label(root, text="", font=("Arial", 12), fg="blue")
# suggestion_label.pack(pady=10)

# capture_button = Button(frame_buttons, text="Capture", command=update_text, font=("Arial", 12))
# capture_button.grid(row=0, column=0, padx=5, pady=5)

# remove_button = Button(frame_buttons, text="Remove Last", command=remove_last, font=("Arial", 12))
# remove_button.grid(row=0, column=1, padx=5, pady=5)

# clear_button = Button(frame_buttons, text="Clear", command=clear_text, font=("Arial", 12))
# clear_button.grid(row=0, column=2, padx=5, pady=5)

# save_button = Button(frame_buttons, text="Save", command=save_text, font=("Arial", 12))
# save_button.grid(row=1, column=0, padx=5, pady=5)

# exit_button = Button(frame_buttons, text="Exit", command=exit_app, font=("Arial", 12))
# exit_button.grid(row=1, column=1, padx=5, pady=5)

# accept_button = tk.Button(root, text="Accept Suggestion", command=accept_suggestion, font=("Arial", 12))
# accept_button.pack(pady=5)

# # Webcam loop
# def webcam_loop():
#     global current_prediction
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(frame_rgb)

#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 # Extract hand landmarks
#                 hand_data = []
#                 for landmark in hand_landmarks.landmark:
#                     hand_data.extend([landmark.x, landmark.y])

#                 # Predict using trained model
#                 hand_data = np.array(hand_data).reshape(1, -1)
#                 current_prediction = model.predict(hand_data)[0]

#                 # Display prediction on webcam
#                 cv2.putText(frame, f"Prediction: {current_prediction}", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 # Draw hand landmarks
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         # Show output
#         cv2.imshow("Hand Sign Recognition", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Start the webcam loop in a separate thread
# threading.Thread(target=webcam_loop, daemon=True).start()

# # Start Tkinter main loop
# root.mainloop()
