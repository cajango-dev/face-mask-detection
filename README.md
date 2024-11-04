# face-mask-detection

______________________________________________

My first project using a webcam like resource, im excited to try, i'm thinking in use python for this project, but idk, my guess is, python is the best and fast language to make something like this.

______________________________________________

# Libraries for a project like this: 

Opencv -> **pip install opencv-python**
Imutils -> **pip install imutils**
Numpy -> **pip install numpy**
TensorFlow -> **pip install tensorflow** (Optional neural network)
Pytorch -> **pip install torch torchvision** (Optional neural network)
Mediapipe -> **pip install mediapipe** (Optional face detection easy way)

______________________________________________

# Project Setup **(Using TensorFlow)**:

face-mask-detection/
│
├── requirements.txt                # List of a project depedencies
├── main.py                         # Main file to run and real-time detect
├── train_model.py                  # Script to train models of mask detection
├── mask_detector.py                # Inf func of the detection model
├── utils.py                        # Aux func for image manipulation
├── models/                         # Directory to save the trained model
│   └── face_mask_model.h5          # Detection model for trained mask
├── dataset/                        # Directory for train data and tests (image of the face with/without mask)
│   ├── with_mask/
│   └── without_mask/
└── README.md                       # Project Documentation

______________________________________________