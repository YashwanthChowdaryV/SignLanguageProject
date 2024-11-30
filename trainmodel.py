import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from function import *  # Assuming the 'function' module contains necessary functions like actions and DATA_PATH

# Label map to convert actions (A-Z) to numeric labels
label_map = {label: num for num, label in enumerate(actions)}

# Containers for sequences and labels
sequences, labels = [], []

# Loop through each action and sequence to load the data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Construct the file path for the .npy file corresponding to this frame
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            
            # Check if the file exists before attempting to load it
            if os.path.exists(file_path):
                res = np.load(file_path)
                window.append(res)
            else:
                print(f"File not found: {file_path}")
                # Handling missing file: add a zeroed array for consistency
                if 'res' in locals():
                    window.append(np.zeros_like(res))  # Append zero array for the missing frame
                else:
                    window.append(np.zeros((63,)))  # If it's the first missing frame, use a default zero array

        # Append the window (sequence) to sequences and the corresponding label to labels
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences and labels into numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets (5% test size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define the log directory for TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))  # Input shape: sequence_length x 63 features per frame
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Output layer size equals the number of actions (A-Z)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print the model summary
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')  # Save the model with weights in H5 format
