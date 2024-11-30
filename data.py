import cv2
import numpy as np
import os
import mediapipe as mp

# Mediapipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Mediapipe Hand Detection Function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False                   # Image is not writable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

# Draw Hand Landmarks
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# Extract Keypoints from Hand Landmarks
def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            keypoints.append(rh)
    return np.concatenate(keypoints) if keypoints else np.zeros(21*3)

# Initialize the actions (A-Z)
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Create directories for actions (A-Z) and sequences
for action in actions:
    for sequence in range(30):  # Number of sequences (videos)
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize Mediapipe Hands Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through actions (A-Z)
    for action in actions:
        # Loop through sequences (videos)
        for sequence in range(30):  # Number of sequences
            # Loop through sequence length (frames)
            for frame_num in range(30):  # Frame length for each sequence
                # Read the frame from the corresponding directory (JPG images now)
                frame_path = f'Image/{action}/{sequence}/{frame_num}.jpg'
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"Skipping missing frame for action {action}, sequence {sequence}, frame {frame_num}")
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)

                # Display 'Starting Collection' message for the first frame of each sequence
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    # Show to screen for the rest of the frames
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Export keypoints to numpy file (.npy)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                # Break gracefully on 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
