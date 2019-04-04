import os
#os.sys.path.append('/home/jovan/.virtualenvs/py3cv4/lib/python3.6/site-packages')
import cv2
from datetime import datetime
import numpy as np
from imutils.video import VideoStream
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras.models import load_model

import time
from skimage.io import imread
from skimage import color
from skimage.transform import resize

DATA = './data'
def scan_folder(parent):
    x,y = [],[]
    # iterate over all the files in directory 'parent'
    for dir_name in os.listdir(parent):
        sub = '/'.join([parent,dir_name])
        if os.path.isdir(sub):
            print(sub)
            for img in os.listdir(sub):
                p = '/'.join([sub,img])
                img = imread(p,as_gray=True)
                img_r = resize(img,(28,28))
                x.append(img_r.copy())
                y.append(ord(dir_name))
    x = np.array(x)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.33)
    X_train = X_train.reshape((len(X_train),28,28,1))
    X_test = X_test.reshape((len(X_test),28,28,1))
    return X_train,y_train,X_test,y_test

def load_MINST():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    return x_train,y_train,x_test,y_test

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x=x_train, y=y_train, epochs=5
)

# y_pred = model.predict_classes(x=x_test)
# print("Test Accuracy: ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=[
#     '%d' % i for i in range(10)
# ], digits=5))

# mnist_dream_path = "./mnist_dream.mp4"

def drawPrediction(img, y_pred, y_class):
    img = 255 - img
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


# cap = VideoStream(src=0).start()
# vw = None
# time.sleep(2)
# while True: # should 481 frames
#     img = cap.read()
#     #if not ret: break

#     #assert img.shape[0] == img.shape[1] # should be a square
#     if img.shape[0] != 720:
#         img = cv2.resize(img, (720, 720))

#     # Pre-process the image for prediction
#     img_proc = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
#     img_proc = cv2.resize(img_proc, (28, 28))
#     img_proc = img_proc / 255.
#     img_proc = 1 - img_proc # inverse since training dataset is white text with black background
#     img_proc = img_proc.reshape((28, 28, 1))
#     img_proc = np.expand_dims(img_proc, axis=0)

#     # Run Prediction
#     y_pred = model.predict_proba(img_proc)[0, :]
#     y_class = np.argmax(y_pred)

#     img = drawPrediction(img, y_pred * 100, y_class)

#     # scale down image for display
#     img_disp = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
#     #cv2_imshow(img_disp)
#     cv2.imshow('output',img_disp)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cap.stop()