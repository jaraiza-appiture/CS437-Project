import os
import numpy as np

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

from tensorflow.python import keras

MODEL_PATH = './cnn.h5'
DATA = './data'

LETTER_MAP = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
              'A' : 10, 'B' : 11, 'C': 12, 'D': 13, 'E': 14, 'F':15,'G':16,
              'H':17, 'I':18, 'i': 19, 'J':20, 'K': 21, 'L': 22, 'l':23,
              'M':24, 'N':25, 'O': 26,'P': 27, 'Q' : 28, 'R': 29, 'S': 30,
              'T': 31, 'U': 32, 'V': 33, 'W': 34, 'X':35, 'Y': 36, 'Z': 37}
REVERSE_LETTER_MAP = {str(v):k for k,v in LETTER_MAP.items()}
MYLIST = []
def plot_classification_report(results):
    x_labels, x_scores = [],[]
    classes = ['%d' % i for i in range(38)]
    ind = [i for i in range(1,39)]
    
    for label,scores in results.items():
        if label in classes:
            x_labels.append(REVERSE_LETTER_MAP[label])
            x_scores.append(scores['precision'])
    plt.figure(figsize=(10, 7))
    plt.bar(ind, x_scores, align='center', alpha=0.5)
    plt.xticks(ind,x_labels)
    plt.ylabel('Precision')
    plt.title('Classification Report')
    
    plt.savefig('gbarplot.png')


def load_NIST(path):
    x,y = [],[]
    for dir_name in os.listdir(path):
        sub = '/'.join([path,dir_name])
        if os.path.isdir(sub):
            for img in os.listdir(sub):
                p = '/'.join([sub,img])
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
    x_train = 1 - x_train # flip to black letters with white backgrounds
    x_test = 1 - x_test
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    return x_train,y_train,x_test,y_test


# Load OCR Data
x_train_NIST,y_train_NIST,x_test_NIST,y_test_NIST = load_NIST(DATA)

# Load MNIST Data
x_train_MNIST,y_train_MNIST,x_test_MNIST,y_test_MNIST = load_MINST()

# Combine and shuffle data
x_train_final = np.concatenate((x_train_NIST, x_train_MNIST), axis=0)
del x_train_NIST # Lower RAM usage
del x_train_MNIST
y_train_final = np.concatenate((y_train_NIST, y_train_MNIST), axis=0)
del y_train_NIST
del y_train_MNIST
x_test_final = np.concatenate((x_test_NIST, x_test_MNIST), axis=0)
del x_test_NIST
del x_test_MNIST
y_test_final = np.concatenate((y_test_NIST, y_test_MNIST), axis=0)
del y_test_NIST
del y_test_MNIST
x_train_final, y_train_final = shuffle(x_train_final, y_train_final)

# Convolutional Neural Network Train
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(38, activation='softmax')
])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(x_train_final, y_train_final, validation_split=0.33, epochs=8)

y_pred = model.predict_classes(x=x_test_final)
print("Test Accuracy: ", accuracy_score(y_test_final, y_pred))
results = classification_report(y_test_final, y_pred, target_names=[
        '%d' % i for i in range(38)], digits=3,output_dict=True) 

print ('Precision:', precision_score(y_test_final, y_pred ,average='micro'))

plot_classification_report(results)
plt.cla()
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('gAccuEpoch.png')

model.save(MODEL_PATH)
