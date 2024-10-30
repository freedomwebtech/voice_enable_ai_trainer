import cv2
import cvzone
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading
import queue
from cvzone.PoseModule import PoseDetector

# Function to get RGB values from mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(0)




# Main video processing loop
while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    img = cv2.resize(img, (1020, 500))
   

    cv2.imshow("RGB", img)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

