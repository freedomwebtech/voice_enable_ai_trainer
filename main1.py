from cvzone.PoseModule import PoseDetector
import cv2
import cvzone
import numpy as np
import pyttsx3
import threading
import queue
import speech_recognition as sr

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
#        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Initialize the webcam
cap = cv2.VideoCapture(0)


# Initialize the PoseDetector class with the given parameters
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

# Function to listen for commands
def listen_for_commands():
    global in_pushup, in_pushup1, pushup_counter, pushup_counter1, is_counting
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("Listening for commands...")

        while True:
            try:
                audio = recognizer.listen(source)  # Listen for commands
                command = recognizer.recognize_google(audio).lower()
                print(f"Command recognized: {command}")

                if "start" in command:
                    print("Starting the counter...")
                    is_counting = True  # Set counting to true
                    in_pushup = False  # Reset push-up state
                    in_pushup1 = False  # Reset alternate push-up state
                    pushup_counter = 0  # Reset counters to zero
                    pushup_counter1 = 0  
                    speak_queue.put("Counters reset to zero and counting started.")  # Notify user via speech

                elif "stop" in command:
                    print("Stopping the counter and resetting...")
                    is_counting = False  # Set counting to false
                    in_pushup = False  # Reset push-up state
                    in_pushup1 = False  # Reset alternate push-up state
                    speak_queue.put("Counters stopped and reset to zero.")
            
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

# Start the command listening thread
#command_thread = threading.Thread(target=listen_for_commands, daemon=True)
#command_thread.start()



# Loop to continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    img = cv2.resize(img, (1020, 500))
    # Find the human pose in the frame
    img = detector.findPose(img)

    # Find the landmarks, bounding box, and center of the body in the frame
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    # Check if any body landmarks are detected
    if lmList:
        # Get coordinates of the landmarks for left push-ups
        cx1, cy1 = lmList[11][0], lmList[11][1]  # Left shoulder
        cx2, cy2 = lmList[13][0], lmList[13][1]  # Left elbow
        cx3, cy3 = lmList[15][0], lmList[15][1]  # Left wrist
        # Get coordinates of the landmarks for right push-ups
        cx4, cy4 = lmList[12][0], lmList[12][1]  # Right shoulder
        cx5, cy5 = lmList[14][0], lmList[14][1]  # Right elbow
        cx6, cy6 = lmList[16][0], lmList[16][1]  # Right wrist

    # Display the image
    cv2.imshow("RGB", img)
    cv2.waitKey(1)
