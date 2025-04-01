# Real-Time-Sign-Language-Recognition

### Overview

This project focuses on real-time recognition of sign language alphabets using computer vision and deep learning techniques. The system captures hand gestures through a webcam, processes them using MediaPipe for hand tracking, and predicts the corresponding alphabet using a trained deep learning model.

### Features

- Real-Time Hand Tracking: Uses MediaPipe to detect and track hand landmarks.

- Alphabet Recognition: A deep learning model predicts sign language alphabets.

- Sentence Formation: Recognized alphabets are stored to form meaningful sentences.

- GUI for Better Usability: Implemented using Tkinter/PyQt to improve user experience.

- Save & Clear Options: Users can save recognized text to a file or clear the input.

- Button-Controlled Predictions: Reduces unwanted predictions by processing input only when a button is clicked.

### Technologies Used

- Python

- OpenCV

- MediaPipe (for hand tracking)

- TensorFlow (for deep learning model)

- Tkinter/PyQt (for GUI)


### Installation

1. Clone the repository:
'''bash  
git clone https://github.com/yourusername/sign-language-recognition.git 
cd sign-language-recognition 

2. Install dependencies:
'''bash
pip install -r requirements.txt 

3. Run the application:
'''bash
python main.py


### How It Works

- The webcam captures hand gestures in real time.

- MediaPipe extracts hand landmarks.

- Preprocessed hand landmarks are fed into a trained deep learning model.

- The model predicts the corresponding alphabet.

- Recognized alphabets are displayed in the GUI and can be stored to form sentences.

### Future Enhancements

- Support for More Sign Languages beyond alphabets.

- Integration with Speech Synthesis for text-to-speech conversion.

- Improved Accuracy using a more advanced deep learning model.
