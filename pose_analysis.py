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

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    rep_count = 0  # Initialize repetition count
    rep_started = False  # Flag to track if a rep is in progres

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties (original width and height)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height

    print(f"Original video resolution: {original_width}x{original_height}")
    print(f"Aspect ratio: {aspect_ratio:.2f}")

    # Define the scaling factor
    scaling_factor = 2  # Adjust this factor to change the output size

    # Calculate new dimensions based on scaling factor
    new_width = original_width // scaling_factor
    new_height = int(new_width / aspect_ratio)  # Maintain aspect ratio

    print(f"Resizing video to: {new_width}x{new_height}")

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            
            if not ret:
                print("encoutered issue with this file")
                break

            # Make detections
            frame = cv2.resize(frame, (new_width, new_height))
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Extract landmarks if pose landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_knee_z = landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].z
                right_knee_z = landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].z
                
                # Choose the side dynamically
                side = "LEFT" if left_knee_z < right_knee_z else "RIGHT"

                #Get the Left landmark attributes
                hip_LEFT = getattr(mp_holistic.PoseLandmark, "LEFT_HIP").value
                knee_LEFT = getattr(mp_holistic.PoseLandmark, "LEFT_KNEE").value
                ankle_LEFT = getattr(mp_holistic.PoseLandmark, "LEFT_ANKLE").value
                
                hip_LEFT_coord = (landmarks[hip_LEFT].x, landmarks[hip_LEFT].y, landmarks[hip_LEFT].z)
                knee_LEFT_coord = (landmarks[knee_LEFT].x, landmarks[knee_LEFT].y, landmarks[knee_LEFT].z)
                ankle_LEFT_coord = (landmarks[ankle_LEFT].x, landmarks[ankle_LEFT].y, landmarks[ankle_LEFT].z)

                #Get the Right landmark attributes
                hip_RIGHT = getattr(mp_holistic.PoseLandmark, "RIGHT_HIP").value
                knee_RIGHT = getattr(mp_holistic.PoseLandmark, "RIGHT_KNEE").value
                ankle_RIGHT = getattr(mp_holistic.PoseLandmark, "RIGHT_ANKLE").value

                # Get coordinates dynamically
                hip_RIGHT_coord = (landmarks[hip_RIGHT].x, landmarks[hip_RIGHT].y, landmarks[hip_RIGHT].z)
                knee_RIGHT_coord = (landmarks[knee_RIGHT].x, landmarks[knee_RIGHT].y, landmarks[knee_RIGHT].z)
                ankle_RIGHT_coord = (landmarks[ankle_RIGHT].x, landmarks[ankle_RIGHT].y, landmarks[ankle_RIGHT].z)
                                # Get coordinates dynamically


                # Calculate knee flexion angle
                knee_angle_RIGHT = calculateAngle3D(hip_RIGHT_coord, knee_RIGHT_coord, ankle_RIGHT_coord)
                knee_angle_LEFT = calculateAngle3D(hip_LEFT_coord, knee_LEFT_coord, ankle_LEFT_coord)
                # angles.append(int(knee_angle))

                # Check for repetition
                if knee_angle_LEFT < 80 and knee_angle_RIGHT < 80 and not rep_started:
                    rep_started = True  # Start the rep
                elif knee_angle_LEFT > 90 and knee_angle_RIGHT > 90 and rep_started:
                    rep_count += 1  # Increment rep count
                    rep_started = False  # Reset for the next rep
                
                
                # Display angle on the image
                cv2.putText(image, f'Right Knee Angle: {int(knee_angle_RIGHT)}',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Left Knee Angle: {int(knee_angle_LEFT)}',
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Reps: {rep_count}',
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # print(angles)
                break
        cap.release()
        cv2.destroyAllWindows()