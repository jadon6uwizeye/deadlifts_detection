import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2

from PIL import Image, ImageTk
from landmarks import landmarks

window = tk.Tk()
window.title("Hand Gesture Recognition")
window.geometry("480x720")
ck.set_appearance_mode("dark")

classLabel =ck.CTkLabel(window, height=1, width=120, text="STAGE", text_color="black")
classLabel.place(x=10, y=1)
classLabel.configure(font=("Arial", 20))

counterLabel = ck.CTkLabel(window, height=1, width=120, text="REPS", text_color="black")
counterLabel.place(x=160, y=1)
counterLabel.configure(font=("Arial", 20))

ProbLable = ck.CTkLabel(window, height=1, width=120, text="PROB", text_color="black" )
ProbLable.place(x=300, y=1)
ProbLable.configure(font=("Arial", 20))

classBox = ck.CTkLabel(window, height=1, width=120,  text_color="black",fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(font=("Arial", 20), text="0")

counteBox = ck.CTkLabel(window, height=1, width=120,  text_color="black", fg_color="blue")
counteBox.place(x=160, y=41)
counteBox.configure(font=("Arial", 20), text="0")

ProbBox = ck.CTkLabel(window, height=1, width=120,  text_color="black", fg_color="blue")
ProbBox.place(x=300, y=41)
ProbBox.configure(font=("Arial", 20), text="0")


def reset_counter():
    global counter
    counter = 0
    counteBox.configure(text=str(counter))

button = ck.CTkButton(window, text="RESET",height=40, width=120, text_color="black", text_font=("Arial", 20), command=reset_counter)
button.place(x=10,y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=100)
lmain = tk.Label(frame)
lmain.place = (0,0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(-1)
current_stage = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class = ''


def detect():
    global current_stage, counter, bodylang_prob, bodylang_class

    ret , frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image)
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    try:
        pass
    except Exception as e:
        pass

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

detect()

window.mainloop()