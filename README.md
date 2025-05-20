# Iranian Sign Language Recognition System (ISLRS)

This project focuses on the real-time recognition of **Iranian Sign Language (ISL)** using computer vision and machine learning techniques. It aims to bridge communication gaps for deaf individuals in Iran by enabling automatic translation of hand gestures into readable or spoken language.

---

##  Project Overview

The goal is to build a system capable of recognizing Iranian sign language letters from live webcam input. The solution uses image processing and a trained machine learning model to classify gestures into 37 sign language classes, corresponding to Persian alphabet letters.

---

##  Project Components

The implementation is modular and consists of the following stages:

1. **Collect Images**  
   Captures images of sign gestures for each class using a webcam and stores them in structured folders.

2. **Save Landmarks**  
   Uses [MediaPipe](https://mediapipe.dev/) to detect and store hand landmarks from the collected images.

3. **Create Dataset**  
   Extracts features (x, y coordinates of 21 hand landmarks) and stores them in a `.csv` file with labels.

4. **Train Network**  
   Trains a `Random Forest` classifier using the dataset for gesture recognition.

5. **Main Application**  
   Loads the trained model and performs real-time gesture recognition via webcam.

---

##  Technologies Used

- **Python**
- **OpenCV** – for image capture and processing.
- **MediaPipe** – for hand tracking and landmark extraction.
- **scikit-learn** – for machine learning model (Random Forest).
- **NumPy / Pandas** – for data manipulation.

---
##  Folder Structure

project/
│
├── images/ # Collected sign language images (37 classes)
├── landmarks/ # Processed images with drawn landmarks
├── dataset.csv # Extracted features and labels
├── model.pkl # Trained Random Forest model
└── main.py # Real-time gesture recognition script
