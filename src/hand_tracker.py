# Import the necessary modules.
import time
import mediapipe as mp
import numpy as np
import cv2 as cv
from utils import draw_landmarks_on_frame

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create an HandLandmarker object with the live stream mode:
latest_result = None

def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result)
with HandLandmarker.create_from_options(options) as landmarker:
    #Load the input video stream from the webcam.
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Empty camera frame.")
            break
        # Convert the frame to RGB format.
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Convert the RGB frame to a MediaPipe's Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Detect hand landmarks from the input video stream.
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        # Process the classification result.
        annotated_frame = draw_landmarks_on_frame(frame, latest_result) if latest_result else frame
    
        # Display the resulting frame
        cv.imshow('frame', annotated_frame)
        if cv.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

