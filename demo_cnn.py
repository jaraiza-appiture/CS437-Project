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
LETTER_MAP = {'A' : 0, 'B' : 1, 'C': 2,'D': 3,
            'E': 4, 'F':5,'G':6, 'H':7, 'I':8,
            'i': 9, 'J':10, 'K': 11, 'L': 12,
            'l':13, 'M':14, 'N':15, 'O': 16,
            'P': 17, 'Q' : 18, 'R': 19, 'S': 20,
            'T': 21,'U': 22, 'V': 23, 'W': 24,
            'X':25, 'Y': 26, 'Z': 27}

REVERSE_LETTER_MAP = {v:k for k,v in LETTER_MAP.items()}

FONT = cv.FONT_HERSHEY_SIMPLEX
SCREEN_TITLE_POS = (10, 500)
BOTTOM_POS = (10,50)
TEXT_COLOR = GREEN
TEXT_SIZE = 1
TEXT_THICK = 2


def write_text(text, image, pos):
    cv.putText(image, text, pos, FONT, TEXT_SIZE,
               BLACK, TEXT_THICK+5, cv.LINE_AA)
    cv.putText(image, text, pos, FONT, TEXT_SIZE,
               TEXT_COLOR, TEXT_THICK, cv.LINE_AA)
    return image


def prediction(img, y_pred, y_class):

    pad_color = 0
    img = np.pad(img, ((0,0), (0,1280-720), (0,0)), mode='constant', constant_values=(pad_color))

    line_type = cv2.LINE_AA
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 2
    x, y = 740, 60
    color = (255, 255, 255)

    text = "Character prediction"
    write(text,img,SCREEN_TITLE_POS)

    text = "Input:"
    cv2.putText(img, text=text, org=(30, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                    color=color, lineType=line_type)
    y = 140
    font_scale = 2.5
    color = (255, 218, 158)
    for i, p in enumerate(y_pred):
        if i == y_class:
            y_classchr = REV_LETTER_MAP[i]
            label = "Pred: %c" % (y_classchr)

            conf = "Conf: %3.2f" % (p) + '%'
            y+=60
            cv2.putText(img, text=label, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
            y+=80
            cv2.putText(img, text=conf, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
            rect_width = 0
            if p > 0: rect_width = int(p * 3.3)
            y+=60
            rect_start = 230
            # cv2.rectangle(img, (x+rect_start, y-5), (x+rect_start+rect_width, y-20), color, -1)

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
    ret,img_proc = cv2.threshold(img_proc,80,255,cv2.THRESH_BINARY)
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

    img = drawPrediction2(img, y_pred * 100, y_class)

    # scale down image for display
    img_disp = cv2.resize(img, None, fx=1, fy=1)

    cv2.imshow('output',img_disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.stop()

