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









import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, Label, Button, filedialog, Frame, PhotoImage
import threading
from PIL import Image, ImageTk
import os

class ModernUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Sign Recognition")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")
        
        # Set theme colors
        self.primary_color = "#3498db"  # Blue
        self.secondary_color = "#2ecc71"  # Green
        self.bg_color = "#f0f0f0"  # Light gray
        self.text_color = "#2c3e50"  # Dark blue/gray
        self.accent_color = "#e74c3c"  # Red
        
        # Variables
        self.recognized_text = ""  # Stores final recognized text
        self.recognized_buffer = ""  # Stores recognized letters for suggestions
        self.current_prediction = ""  # Stores the current hand sign prediction
        self.common_words = ["hello", "help", "world", "good", "morning", "thank", "you", "please"]
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load trained model
        self.model = joblib.load("hand_sign_model.pkl")
        
        # Start webcam
        self.cap = cv2.VideoCapture(0)
        
        # Create custom style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure button styles
        self.style.configure('Primary.TButton', 
                            background=self.primary_color, 
                            foreground='white', 
                            font=('Arial', 12, 'bold'),
                            padding=10,
                            borderwidth=0)
        
        self.style.configure('Secondary.TButton', 
                            background=self.secondary_color, 
                            foreground='white', 
                            font=('Arial', 12, 'bold'),
                            padding=10,
                            borderwidth=0)
        
        self.style.configure('Danger.TButton', 
                            background=self.accent_color, 
                            foreground='white', 
                            font=('Arial', 12, 'bold'),
                            padding=10,
                            borderwidth=0)
        
        self.style.map('Primary.TButton', 
                      background=[('active', '#2980b9')])
        
        self.style.map('Secondary.TButton', 
                      background=[('active', '#27ae60')])
        
        self.style.map('Danger.TButton', 
                      background=[('active', '#c0392b')])
        
        # Create main frames
        self.create_frames()
        
        # Create UI elements
        self.create_header()
        self.create_video_frame()
        self.create_text_display()
        self.create_suggestion_area()
        self.create_control_buttons()
        
        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
        self.webcam_thread.start()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_frames(self):
        # Main container with padding
        self.main_container = Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for video
        self.left_panel = Frame(self.main_container, bg=self.bg_color)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for controls and text
        self.right_panel = Frame(self.main_container, bg=self.bg_color)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
    
    def create_header(self):
        header_frame = Frame(self.left_panel, bg=self.primary_color, height=60)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = Label(header_frame, 
                           text="Hand Sign Recognition", 
                           font=("Arial", 18, "bold"), 
                           fg="white", 
                           bg=self.primary_color)
        title_label.pack(pady=10)
    
    def create_video_frame(self):
        self.video_container = Frame(self.left_panel, bg="black", width=480, height=360)
        self.video_container.pack(pady=10)
        
        # Make sure the frame keeps its size
        self.video_container.pack_propagate(False)
        
        # Label to display webcam feed
        self.video_label = Label(self.video_container, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Current prediction display
        self.prediction_frame = Frame(self.left_panel, bg=self.bg_color)
        self.prediction_frame.pack(fill=tk.X, pady=10)
        
        self.prediction_label = Label(self.prediction_frame, 
                                     text="Current Sign: None", 
                                     font=("Arial", 14, "bold"), 
                                     fg=self.text_color, 
                                     bg=self.bg_color)
        self.prediction_label.pack(pady=5)
    
    def create_text_display(self):
        text_frame = Frame(self.right_panel, bg=self.bg_color)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        text_header = Label(text_frame, 
                           text="Recognized Text", 
                           font=("Arial", 14, "bold"), 
                           fg=self.text_color, 
                           bg=self.bg_color)
        text_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Text display with border and styling
        self.text_display = tk.Text(text_frame, 
                                   font=("Arial", 12), 
                                   height=10, 
                                   wrap=tk.WORD, 
                                   bg="white", 
                                   fg=self.text_color,
                                   padx=10,
                                   pady=10,
                                   relief=tk.SOLID,
                                   borderwidth=1)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        self.text_display.config(state=tk.DISABLED)  # Make it read-only
    
    def create_suggestion_area(self):
        suggestion_frame = Frame(self.right_panel, bg=self.bg_color)
        suggestion_frame.pack(fill=tk.X, pady=10)
        
        suggestion_header = Label(suggestion_frame, 
                                 text="Suggestions", 
                                 font=("Arial", 14, "bold"), 
                                 fg=self.text_color, 
                                 bg=self.bg_color)
        suggestion_header.pack(anchor=tk.W, pady=(0, 5))
        
        self.suggestion_label = Label(suggestion_frame, 
                                     text="", 
                                     font=("Arial", 12), 
                                     fg=self.primary_color, 
                                     bg=self.bg_color)
        self.suggestion_label.pack(anchor=tk.W, pady=5)
        
        self.accept_button = ttk.Button(suggestion_frame, 
                                       text="Accept Suggestion", 
                                       command=self.accept_suggestion,
                                       style='Secondary.TButton')
        self.accept_button.pack(anchor=tk.W, pady=5)
    
    def create_control_buttons(self):
        button_frame = Frame(self.right_panel, bg=self.bg_color)
        button_frame.pack(fill=tk.X, pady=10)
        
        # First row of buttons
        row1 = Frame(button_frame, bg=self.bg_color)
        row1.pack(fill=tk.X, pady=5)
        
        self.capture_button = ttk.Button(row1, 
                                        text="Capture", 
                                        command=self.update_text,
                                        style='Primary.TButton')
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        self.remove_button = ttk.Button(row1, 
                                       text="Remove Last", 
                                       command=self.remove_last,
                                       style='Primary.TButton')
        self.remove_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(row1, 
                                      text="Clear", 
                                      command=self.clear_text,
                                      style='Danger.TButton')
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Second row of buttons
        row2 = Frame(button_frame, bg=self.bg_color)
        row2.pack(fill=tk.X, pady=5)
        
        self.save_button = ttk.Button(row2, 
                                     text="Save", 
                                     command=self.save_text,
                                     style='Secondary.TButton')
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(row2, 
                                     text="Exit", 
                                     command=self.exit_app,
                                     style='Danger.TButton')
        self.exit_button.pack(side=tk.LEFT, padx=5)
    
    def update_text(self):
        if self.current_prediction:  # Only update if there's a valid prediction
            self.recognized_buffer += self.current_prediction  # Append the predicted letter to the buffer
            self.recognized_text += self.current_prediction  # Append the predicted letter to the final text
            
            # Update the text display
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.recognized_text)
            self.text_display.config(state=tk.DISABLED)
            
            # Update suggestion
            suggestion = self.suggest_word(self.recognized_buffer)
            if suggestion:
                self.suggestion_label.config(text=f"Suggestion: {suggestion}")
            else:
                self.suggestion_label.config(text="")  # Clear suggestion if none
            
            # Update status
            self.status_var.set(f"Added character: {self.current_prediction}")
    
    def suggest_word(self, buffer):
        matches = [word for word in self.common_words if word.startswith(buffer)]
        return matches[0] if matches else None  # Return first match or None
    
    def accept_suggestion(self):
        suggestion = self.suggest_word(self.recognized_buffer)
        if suggestion:
            # Replace the buffer with the suggestion in the final text
            self.recognized_text = self.recognized_text[:-len(self.recognized_buffer)] + suggestion + " "
            
            # Update the text display
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.recognized_text)
            self.text_display.config(state=tk.DISABLED)
            
            self.recognized_buffer = ""  # Clear the buffer
            self.suggestion_label.config(text="")  # Clear the suggestion
            
            # Update status
            self.status_var.set(f"Accepted suggestion: {suggestion}")
    
    def remove_last(self):
        if self.recognized_text:
            self.recognized_text = self.recognized_text[:-1]  # Remove the last character
            self.recognized_buffer = self.recognized_buffer[:-1]  # Remove the last character from the buffer
            
            # Update the text display
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, self.recognized_text)
            self.text_display.config(state=tk.DISABLED)
            
            # Update suggestion
            suggestion = self.suggest_word(self.recognized_buffer)
            if suggestion:
                self.suggestion_label.config(text=f"Suggestion: {suggestion}")
            else:
                self.suggestion_label.config(text="")  # Clear suggestion if none
            
            # Update status
            self.status_var.set("Removed last character")
    
    def clear_text(self):
        self.recognized_text = ""
        self.recognized_buffer = ""
        
        # Update the text display
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)
        self.text_display.config(state=tk.DISABLED)
        
        self.suggestion_label.config(text="")  # Clear suggestion
        
        # Update status
        self.status_var.set("Text cleared")
    
    def save_text(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            with open(file_path, "w") as file:
                file.write(self.recognized_text)
            
            # Update status
            self.status_var.set(f"Text saved to {os.path.basename(file_path)}")
    
    def exit_app(self):
        self.root.quit()  # Stop Tkinter main loop
        self.cap.release()  # Release webcam
        cv2.destroyAllWindows()  # Close OpenCV windows
    
    def webcam_loop(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(frame_rgb)
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Add a title bar
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), (52, 152, 219), -1)
            cv2.putText(display_frame, "Hand Sign Recognition", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Extract hand landmarks
                    hand_data = []
                    for landmark in hand_landmarks.landmark:
                        hand_data.extend([landmark.x, landmark.y])
                    
                    # Predict using trained model
                    hand_data = np.array(hand_data).reshape(1, -1)
                    self.current_prediction = self.model.predict(hand_data)[0]
                    
                    # Update prediction label
                    self.prediction_label.config(text=f"Current Sign: {self.current_prediction}")
                    
                    # Display prediction on webcam with a nicer box
                    cv2.rectangle(display_frame, (10, display_frame.shape[0] - 60), 
                                 (300, display_frame.shape[0] - 10), (46, 204, 113), -1)
                    cv2.putText(display_frame, f"Prediction: {self.current_prediction}", 
                               (20, display_frame.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Draw hand landmarks with custom styling
                    self.mp_draw.draw_landmarks(
                        display_frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(52, 152, 219), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(46, 204, 113), thickness=2)
                    )
            
            # Convert to PIL format for Tkinter
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((480, 360))
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update video label
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk  # Keep a reference
            
            # Process Tkinter events to keep UI responsive
            self.root.update_idletasks()
            self.root.update()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernUI(root)
    root.mainloop()



