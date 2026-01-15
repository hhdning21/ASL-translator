import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

try:
    detector = mp_hands.Hands()
    print("Success! MediaPipe is working correctly.")
except Exception as e:
    print(f"Still hitting an error: {e}")