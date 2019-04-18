import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python import keras
from skimage.io import imread
from skimage.transform import resize

MODEL_PATH = './cnn.h5'
DATA = './data'
LETTER_MAP = {'A' : 0, 'B' : 1, 'C': 2, 'D': 3, 'E': 4, 'F':5,'G':6, 'H':7, 'I':8, 'i': 9, 'J':10, 'K': 11, 'L': 12, 'l':13, 'M':14, 'N':15, 'O': 16,
'P': 17, 'Q' : 18, 'R': 19, 'S': 20, 'T': 21, 'U': 22, 'V': 23, 'W': 24, 'X':25, 'Y': 26, 'Z': 27}
def load_NIST(path):
    x,y = [],[]
    # print ("path: ", path)

    for dir_name in os.listdir(path):
        # print ("dir_name: ", dir_name)

        sub = '/'.join([path,dir_name])
        # print ("sub: ", sub)

        if os.path.isdir(sub):
            # print(sub)

            for img in os.listdir(sub):
                p = '/'.join([sub,img])
                # print("p is: ", p)
                img = imread(p,as_gray=True)
                img_r = resize(img,(28,28))
                x.append(img_r.copy())
                y.append(LETTER_MAP[dir_name])
    x = np.array(x)
    y = np.array(y)
    x, y = shuffle(x, y)
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

x_train_NIST,y_train_NIST,x_test_NIST,y_test_NIST = load_NIST(DATA)
# x_train_MNIST,y_train_MNIST,x_test_MNIST,y_test_MNIST = load_MINST()

# x_train_final = np.concatenate((x_train_NIST, x_train_MNIST), axis=0)
# del x_train_NIST
# del x_train_MNIST
# y_train_final = np.concatenate((y_train_NIST, y_train_MNIST), axis=0)
# del y_train_NIST
# del y_train_MNIST
# x_test_final = np.concatenate((x_test_NIST, x_test_MNIST), axis=0)
# del x_test_NIST
# del x_test_MNIST
# y_test_final = np.concatenate((y_test_NIST, y_test_MNIST), axis=0)
# del y_test_NIST
# del y_test_MNIST
# x_train_final, y_train_final = shuffle(x_train_final, y_train_final)

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    # keras.layers.MaxPooling2D((2, 2)),
    # keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(28, activation='softmax')
])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x=x_train_NIST, y=y_train_NIST, epochs=8
)

y_pred = model.predict_classes(x=x_test_NIST)
print("Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
print(classification_report(y_test_NIST, y_pred, target_names=[
    '%d' % i for i in range(28)
], digits=3))

model.save(MODEL_PATH)
