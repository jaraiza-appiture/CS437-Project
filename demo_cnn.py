import os
os.sys.path.append('/home/jovan/.virtualenvs/py3cv4/lib/python3.6/site-packages')
import cv2
from datetime import datetime
import numpy as np
from imutils.video import VideoStream
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
import time

MODEL_PATH = './cnn.h5'

def drawPrediction(img, y_pred, y_class):
    #img = 255 - img
    pad_color = 0
    img = np.pad(img, ((0,0), (0,1280-720), (0,0)), mode='constant', constant_values=(pad_color))

    line_type = cv2.LINE_AA
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 2
    x, y = 740, 60
    color = (255, 255, 255)

    text = "Neural Network Output:"
    cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                    color=color, lineType=line_type)

    text = "Input:"
    cv2.putText(img, text=text, org=(30, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                    color=color, lineType=line_type)
    y = 130
    for i, p in enumerate(y_pred):
        if i == y_class: color = (255, 218, 158)
        else: color = (100, 100, 100)

        rect_width = 0
        if p > 0: rect_width = int(p * 3.3)

        rect_start = 230
        cv2.rectangle(img, (x+rect_start, y-5), (x+rect_start+rect_width, y-20), color, -1)

        text = "%d: %3.2f" % (i, p)
        cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                    color=color, lineType=line_type)
        y += 60
    return img

model = load_model(MODEL_PATH) # load pre-trained cnn
cap = VideoStream(src=0).start() # start cam
time.sleep(2) # let cam warm up

while True:
    img = cap.read()
    #if not ret: break

    #assert img.shape[0] == img.shape[1] # should be a square
    if img.shape[0] != 720:
        img = cv2.resize(img, (720, 720))

    # Pre-process the image for prediction
    img_proc = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    #img_proc = cv2.resize(img_proc, (28, 28),interpolation=cv2.INTER_AREA)
    ret,img_proc = cv2.threshold(img_proc,80,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh',img_proc)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    img_proc = cv2.resize(img_proc, (28, 28),interpolation=cv2.INTER_AREA)
    img_proc = img_proc / 255.
    #img_proc = 1 - img_proc # inverse since training dataset is white text with black background
    img_proc = img_proc.reshape((28, 28, 1))
    img_proc = np.expand_dims(img_proc, axis=0)

    # Run Prediction
    y_pred = model.predict_proba(img_proc)[0, :]
    y_class = np.argmax(y_pred)

    img = drawPrediction(img, y_pred * 100, y_class)

    # scale down image for display
    img_disp = cv2.resize(img, None, fx=1, fy=1)

    cv2.imshow('output',img_disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.stop()

