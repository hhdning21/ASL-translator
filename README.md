# ASL-translator
Real-time American Sign Language (ASL) translator using MediaPipe hand tracking and machine learning.

## Features
- Real-time hand gesture recognition via webcam
- MediaPipe hand landmark detection
- Random Forest classifier for sign prediction
- Supports multiple ASL signs 


### Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe scikit-learn numpy
```

## Data Folder Structure

The `data/` folder is **not included** in this repository (excluded via `.gitignore`). You need to create it and collect your own training data:

```
data/
├── 0/          # Images for sign 'A'
├── 1/          # Images for sign 'B'
├── 2/          # Images for sign 'C'
└── ...         # Add more folders for additional signs
```
