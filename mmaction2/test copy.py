import cv2
import numpy as np
import os
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.LEFT_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw LEFT hand connections
    mp_drawing.draw_landmarks(image, results.RIGHT_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw LEFT hand connections

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

def calculateAngle3D(l1, l2, l3):
    # Convert landmarks to vectors
    vector1 = np.array([l1[0] - l2[0], l1[1] - l2[1], l1[2] - l2[2]])
    vector2 = np.array([l3[0] - l2[0], l3[1] - l2[1], l3[2] - l2[2]])

    # Normalize the vectors
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the angle between the vectors
    angle = np.arccos(np.dot(vector1, vector2))
    return np.degrees(angle)

cap = cv2.VideoCapture("demo\dennis.mp4")
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Extract landmarks if pose landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_knee_z = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].z
            right_knee_z = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].z
            
            # Choose the side dynamically
            side = "LEFT" if left_knee_z < right_knee_z else "RIGHT"

            # Dynamically get the landmark attributes
            hip = getattr(mp_holistic.PoseLandmark, f"{side}_HIP").value
            knee = getattr(mp_holistic.PoseLandmark, f"{side}_KNEE").value
            ankle = getattr(mp_holistic.PoseLandmark, f"{side}_ANKLE").value

            # Get coordinates dynamically
            hip_coord = (landmarks[hip].x, landmarks[hip].y, landmarks[hip].z)
            knee_coord = (landmarks[knee].x, landmarks[knee].y, landmarks[knee].z)
            ankle_coord = (landmarks[ankle].x, landmarks[ankle].y, landmarks[ankle].z)

            # Calculate knee flexion angle
            knee_angle = calculateAngle3D(hip_coord, knee_coord, ankle_coord)
            
            # Display angle on the image
            cv2.putText(image, f'{side} Knee Angle: {int(knee_angle)}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()