import os
import csv
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './images'
CSV_FILE = 'data.csv'

# Define a fixed size for the data entries
FIXED_SIZE = 42  # Assuming 21 landmarks with x and y coordinates

data = []
labels = []

# Iterate over each subdirectory in the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip if not a directory

    # Iterate over each image in the subdirectory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Ensure data_aux has a fixed size
            if len(data_aux) < FIXED_SIZE:
                data_aux.extend([0] * (FIXED_SIZE - len(data_aux)))
            elif len(data_aux) > FIXED_SIZE:
                data_aux = data_aux[:FIXED_SIZE]

            data.append(data_aux)
            labels.append(dir_)

# Save the data and labels to a CSV file
with open(CSV_FILE, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    header = [f'feature_{i}' for i in range(FIXED_SIZE)] + ['label']
    csvwriter.writerow(header)
    # Write the data
    for features, label in zip(data, labels):
        csvwriter.writerow(features + [label])