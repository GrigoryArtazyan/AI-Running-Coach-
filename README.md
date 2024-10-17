# AI-Running-Coach-
AI Running Coach: Based on MoveNet Refined


# This project is a webcam-based AI running coach that uses TensorFlow and OpenCV for keypoint detection. It provides real-time feedback on running posture and form.

## Features
- Real-time keypoint detection using TensorFlow and OpenCV.
- AI-based posture correction and movement analysis.
- Web interface for ease of use.

## Set Up

1. Install and Import Dependencies
    ```bash
    pip install --upgrade pip
    ```
    also CPU based model
   ```bash
    !pip install tensorflow==2.4.1 opencv-python matplotlib
    ```
   including
   ```bash
      import tensorflow as tf
      import numpy as np
      from matplotlib import pyplot as plt
      import cv2
    ```
   

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python run_coach.py
    ```

## Usage

To start the webcam-based AI coach, simply

