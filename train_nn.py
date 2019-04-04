import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from skimage.io import imread
from skimage.transform import resize

MODEL_PATH = './cnn.h5'
DATA = './data'

def load_NIST(path):
    x,y = [],[]

    for dir_name in os.listdir(path):
        sub = '/'.join([path,dir_name])
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

x_train,y_train,x_test,y_test = load_MINST()

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

y_pred = model.predict_classes(x=x_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=[
    '%d' % i for i in range(10)
], digits=5))

model.save(MODEL_PATH)
