import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Make image not writable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Make image writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            keypoints.append(rh)
    return np.concatenate(keypoints) if keypoints else np.zeros(21*3)

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Create an array of actions from 'A' to 'Z'
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Number of sequences per action and the length of each sequence
no_sequences = 30
sequence_length = 30

# Create directories for A-Z actions if they don't exist
for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# Initialize mediapipe hands model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    cap = cv2.VideoCapture(0)  # Start the webcam
    sequence_data = []
    current_action = 'A'  # Start with 'A', you can change this dynamically
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, hands)  # Detect hands
        draw_styled_landmarks(image, results)  # Draw hand landmarks
        
        keypoints = extract_keypoints(results)  # Extract the keypoints

        # Add keypoints to the sequence data
        sequence_data.append(keypoints)
        
        # Show the webcam feed with landmarks and current action
        cv2.putText(image, f"Current Action: {current_action}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Landmarks", image)
        
        if len(sequence_data) == sequence_length:
            # When the sequence is complete, save the data
            filename = os.path.join(DATA_PATH, current_action, f"{len(os.listdir(os.path.join(DATA_PATH, current_action)))}.jpg")
            cv2.imwrite(filename, frame)  # Save the frame as JPG

            # Reset the sequence data for the next one
            sequence_data = []

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == ord('q'):  # Quit the loop when 'q' is pressed
            break
        elif interrupt & 0xFF == ord('a'):  # Change action to 'A' when 'a' is pressed
            current_action = 'A'
        elif interrupt & 0xFF == ord('b'):  # Change action to 'B' when 'b' is pressed
            current_action = 'B'
        elif interrupt & 0xFF == ord('c'):  # Change action to 'C' when 'c' is pressed
            current_action = 'C'
        elif interrupt & 0xFF == ord('d'):  # Change action to 'D' when 'd' is pressed
            current_action = 'D'
        elif interrupt & 0xFF == ord('e'):  # Change action to 'E' when 'e' is pressed
            current_action = 'E'
        elif interrupt & 0xFF == ord('f'):  # Change action to 'F' when 'f' is pressed
            current_action = 'F'
        elif interrupt & 0xFF == ord('g'):  # Change action to 'G' when 'g' is pressed
            current_action = 'G'
        elif interrupt & 0xFF == ord('h'):  # Change action to 'H' when 'h' is pressed
            current_action = 'H'
        elif interrupt & 0xFF == ord('i'):  # Change action to 'I' when 'i' is pressed
            current_action = 'I'
        elif interrupt & 0xFF == ord('j'):  # Change action to 'J' when 'j' is pressed
            current_action = 'J'
        elif interrupt & 0xFF == ord('k'):  # Change action to 'K' when 'k' is pressed
            current_action = 'K'
        elif interrupt & 0xFF == ord('l'):  # Change action to 'L' when 'l' is pressed
            current_action = 'L'
        elif interrupt & 0xFF == ord('m'):  # Change action to 'M' when 'm' is pressed
            current_action = 'M'
        elif interrupt & 0xFF == ord('n'):  # Change action to 'N' when 'n' is pressed
            current_action = 'N'
        elif interrupt & 0xFF == ord('o'):  # Change action to 'O' when 'o' is pressed
            current_action = 'O'
        elif interrupt & 0xFF == ord('p'):  # Change action to 'P' when 'p' is pressed
            current_action = 'P'
        elif interrupt & 0xFF == ord('q'):  # Change action to 'Q' when 'q' is pressed
            current_action = 'Q'
        elif interrupt & 0xFF == ord('r'):  # Change action to 'R' when 'r' is pressed
            current_action = 'R'
        elif interrupt & 0xFF == ord('s'):  # Change action to 'S' when 's' is pressed
            current_action = 'S'
        elif interrupt & 0xFF == ord('t'):  # Change action to 'T' when 't' is pressed
            current_action = 'T'
        elif interrupt & 0xFF == ord('u'):  # Change action to 'U' when 'u' is pressed
            current_action = 'U'
        elif interrupt & 0xFF == ord('v'):  # Change action to 'V' when 'v' is pressed
            current_action = 'V'
        elif interrupt & 0xFF == ord('w'):  # Change action to 'W' when 'w' is pressed
            current_action = 'W'
        elif interrupt & 0xFF == ord('x'):  # Change action to 'X' when 'x' is pressed
            current_action = 'X'
        elif interrupt & 0xFF == ord('y'):  # Change action to 'Y' when 'y' is pressed
            current_action = 'Y'
        elif interrupt & 0xFF == ord('z'):  # Change action to 'Z' when 'z' is pressed
            current_action = 'Z'

    cap.release()
    cv2.destroyAllWindows()
