import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import cvzone
import speech_recognition as sr
import time
# Initialize the YOLO model and video capture
model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

# Initialize variables
count = 0




# Main loop for video capture and processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    count += 1
    if count % 2 != 0:
        continue
    
    # Make predictions
    result = model.track(frame)
    
    if result[0].boxes is not None and result[0].boxes.id is not None:
        keypoints = result[0].keypoints.xy.cpu().numpy()
        


    # Display the frame
    cv2.imshow("RGB", frame)

    # Exit on 'Esc' key press
    if cv2.waitKey(1)&0xFF==27: 
        break

# Release resources
cap.release()
cv2.destroyAllWindows()  
