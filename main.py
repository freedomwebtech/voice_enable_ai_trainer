import cv2
import numpy as np
import cvzone
import speech_recognition as sr
import pyttsx3
import threading
from queue import Queue
from ultralytics import YOLO

class PushUpCounter:
    def __init__(self, video_source=0, model_path='yolo11n-pose.pt'):
        self.cap = cv2.VideoCapture(video_source)
        self.running = True

    def run(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (1020, 500))
            cv2.imshow('Push Up Counter', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False
        self.speech_thread.join()  # Ensure the speech thread finishes

if __name__ == "__main__":
    counter = PushUpCounter()
    counter.run()
