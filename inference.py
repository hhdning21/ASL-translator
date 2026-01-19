import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Map folder names to actual letters
labels_dict = {str(i): chr(65+i) for i in range(26)}  # 0='A', 1='B', ..., 25='Z'

while True:
    ret, frame = cap.read()
    if not ret: break
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand skeleton for the UI
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data_aux = []
            x_ = []
            y_ = []

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

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[prediction[0]]

            # Display logic
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x1+100, y1-50), (0, 255, 0), -1)
            cv2.putText(frame, predicted_character, (x1+20, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('ASL Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()