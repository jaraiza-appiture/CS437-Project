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

LETTER_MAP = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
              'A' : 10, 'B' : 11, 'C': 12, 'D': 13, 'E': 14, 'F':15,'G':16,
              'H':17, 'I':18, 'i': 19, 'J':20, 'K': 21, 'L': 22, 'l':23,
              'M':24, 'N':25, 'O': 26,'P': 27, 'Q' : 28, 'R': 29, 'S': 30,
              'T': 31, 'U': 32, 'V': 33, 'W': 34, 'X':35, 'Y': 36, 'Z': 37}
REVERSE_LETTER_MAP = {v:k for k,v in LETTER_MAP.items()}
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 50, 50)
BLACK = (0,0,0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PREDPOS = (740, 50)
INPUTPOS = (30,50)
TEXT_COLOR = GREEN
TEXT_SIZE = 1
TEXT_THICK = 2


def write_text(text, image, pos):
    cv2.putText(image, text, pos, FONT, TEXT_SIZE,
               BLACK, TEXT_THICK+5, cv2.LINE_AA)
    cv2.putText(image, text, pos, FONT, TEXT_SIZE,
               TEXT_COLOR, TEXT_THICK, cv2.LINE_AA)
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

    text = "Character Prediction:"
    write_text(text,img,PREDPOS)

    text = "Input:"
    write_text(text,img,INPUTPOS)

    y = 140
    font_scale = 2.5
    color = (255, 218, 158)
    for i, p in enumerate(y_pred):
        if i == y_class:
            y_classchr = REVERSE_LETTER_MAP[i]
            label = "Pred: %c" % (y_classchr)

            conf = "Conf: %3.2f" % (p) + '%'
            y+=60
            write_text(label,img,(x,y))
            # cv2.putText(img, text=label, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
            #             color=color, lineType=line_type)
            y+=80
            write_text(conf,img,(x,y))
            # cv2.putText(img, text=conf, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
            #             color=color, lineType=line_type)
    return img
if __name__ == '__main__':
    model = load_model(MODEL_PATH) # load pre-trained cnn
    cap = VideoStream(src=0).start() # start cam
    time.sleep(2) # let cam warm up

    while True:

        img = cap.read()

        img = cv2.resize(img, (720, 720))

        # Pre-process the image for prediction
        img_proc = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        ret,img_proc = cv2.threshold(img_proc,80,255,cv2.THRESH_BINARY)

        cv2.imshow('thresh',img_proc)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        img_proc = cv2.resize(img_proc, (28, 28),interpolation=cv2.INTER_AREA)
        img_proc = img_proc / 255.
        img_proc = img_proc.reshape((28, 28, 1))
        img_proc = np.expand_dims(img_proc, axis=0)

        # Run Prediction
        y_pred = model.predict_proba(img_proc)[0, :]
        y_class = np.argmax(y_pred)

        img = prediction(img, y_pred * 100, y_class)

        cv2.imshow('output',img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.stop()

