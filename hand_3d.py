# This below code is the 3D GUI Hand Sign Dectetor Code 

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Set up the 3D plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initial empty scatter plot
scatter = ax.scatter([], [], [], c='r', marker='o')

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 3D coordinates
                landmarks_3d = []
                for lm in hand_landmarks.landmark:
                    landmarks_3d.append([lm.x, lm.y, lm.z])

                landmarks_3d = np.array(landmarks_3d)  # Convert to NumPy array

                # Update the 3D plot with new landmarks
                scatter._offsets3d = (landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])

                # Draw the plot again with updated landmarks
                plt.draw()
                plt.pause(0.01)

        cv2.imshow("3D Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()





# import cv2
# import mediapipe as mp
# import numpy as np
# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# cap = cv2.VideoCapture(0)

# # Global variables
# landmarks_3d = []
# alpha = 0.2  # Smoothing factor
# prev_landmarks = None  # Store previous frame landmarks

# def smooth_landmarks(new_landmarks):
#     """Applies Exponential Moving Average (EMA) smoothing to hand landmarks."""
#     global prev_landmarks
#     new_landmarks = np.array(new_landmarks)

#     if prev_landmarks is None:
#         prev_landmarks = new_landmarks  # Initialize with first frame
#     else:
#         prev_landmarks = alpha * new_landmarks + (1 - alpha) * prev_landmarks  # Apply EMA
    
#     return prev_landmarks.tolist()

# def draw_hand():
#     """OpenGL function to render 3D hand landmarks."""
#     global landmarks_3d
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glLoadIdentity()

#     if not landmarks_3d or len(landmarks_3d) < 21:
#         print("No hand landmarks detected")
#         glutSwapBuffers()
#         return

#     # Adjusting camera position to center the hand
#     glTranslatef(0.0, 0.0, -2.5)  # Move the hand into view

#     glColor3f(1.0, 0.0, 0.0)
#     glPointSize(5)
    
#     glBegin(GL_POINTS)
#     for point in landmarks_3d:
#         glVertex3f(point[0], -point[1], -point[2])  # Flip Y-axis for correct orientation
#     glEnd()

#     glutSwapBuffers()

# def update_landmarks():
#     """Captures frame, processes hand landmarks, and applies smoothing."""
#     global landmarks_3d
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     if result.multi_hand_landmarks:
#         hand_landmarks = result.multi_hand_landmarks[0]
#         new_landmarks = [(lm.x - 0.5, lm.y - 0.5, lm.z) for lm in hand_landmarks.landmark]
#         landmarks_3d = smooth_landmarks(new_landmarks)  # Apply smoothing
    
#     glutPostRedisplay()

# def opengl_setup():
#     """Initializes OpenGL settings."""
#     glutInit()
#     glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
#     glutInitWindowSize(800, 600)
#     glutCreateWindow(b"3D Hand Model")

#     glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D effect

#     glMatrixMode(GL_PROJECTION)
#     glLoadIdentity()
#     gluPerspective(45, 1, 0.1, 10)  # Improved field of view
#     glMatrixMode(GL_MODELVIEW)

#     glutDisplayFunc(draw_hand)
#     glutIdleFunc(update_landmarks)
#     glutMainLoop()

# if __name__ == "__main__":
#     opengl_setup()


