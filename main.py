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


# Initialize the PoseDetector class
detector = PoseDetector(staticMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False, 
                        smoothSegmentation=True, detectionCon=0.5, trackCon=0.5)

# Speech recognition function
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        if mode == 'stop':  # Check if the mode is 'stop'
            print("Stopping speech recognition.")
            break  # Exit the loop if mode is 'stop'
        
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source, timeout=1)  # Shortened timeout for responsiveness
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                if "normal" in command:
                    speak("Starting both mode")
                    set_mode('both')
                elif "combine" in command:
                    speak("Starting combine mode")
                    set_mode('combine')
                elif "stop" in command:
                    speak("Quitting application")
                    set_mode('stop')
                    
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.WaitTimeoutError:
            continue  # Ignore timeout errors and continue listening





# Run speech recognition in a separate thread
#command_thread = threading.Thread(target=listen_for_commands, daemon=True)
#command_thread.start()

# Main video processing loop
while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    img = cv2.resize(img, (1020, 500))
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    if lmList:
        cx1, cy1 = lmList[11][0], lmList[11][1]  # Left shoulder
        cx2, cy2 = lmList[13][0], lmList[13][1]  # Left elbow
        cx3, cy3 = lmList[15][0], lmList[15][1]  # Left wrist
        cx4, cy4 = lmList[12][0], lmList[12][1]  # Right shoulder
        cx5, cy5 = lmList[14][0], lmList[14][1]  # Right elbow
        cx6, cy6 = lmList[16][0], lmList[16][1]  # Right wrist

        

    cv2.imshow("RGB", img)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
