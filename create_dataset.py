import os
import mediapipe as mp
import cv2
import pickle
import numpy as np

# Import MediaPipe hand detection
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'
data = []
labels = []

if not os.path.exists(DATA_DIR):
    print("Error: 'data' folder not found. Run collect_data.py first!")
    exit()

for dir_ in os.listdir(DATA_DIR):
    if dir_.startswith('.') or not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue 
    
    print(f'Processing class: {dir_}')
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if img_path.startswith('.'): continue
        
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # We only care about the first hand found in the photo
            hand_landmarks = results.multi_hand_landmarks[0]
            
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

            data.append(data_aux)
            labels.append(dir_)

# Save the extracted features
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Finished! Extracted landmarks from {len(data)} images.")