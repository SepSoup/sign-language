import os
import cv2
import mediapipe as mp

DATA_DIR = './images'
OUTPUT_DIR = './LM_OUTPUTS'

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iterate over each class directory
for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    # Process only the first image in each class directory
    image_name = os.listdir(class_dir)[0]
    image_path = os.path.join(class_dir, image_name)
    image = cv2.imread(image_path)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    # Check if landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save the image with landmarks
    output_image_path = os.path.join(OUTPUT_DIR, f'{label}.jpg')
    cv2.imwrite(output_image_path, image)

# Release resources
hands.close() 