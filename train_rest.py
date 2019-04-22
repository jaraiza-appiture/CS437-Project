import os
import numpy as np

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import GaussianNB

from skimage.io import imread
from skimage.transform import resize

MODEL_PATH = './cnn.h5'
DATA = './data'
LETTER_MAP = {'A' : 0, 'B' : 1, 'C': 2, 'D': 3,
              'E': 4, 'F':5,'G':6, 'H':7, 'I':8,
              'i': 9, 'J':10, 'K': 11, 'L': 12,
              'l':13, 'M':14, 'N':15, 'O': 16,
              'P': 17, 'Q' : 18, 'R': 19, 'S': 20,
              'T': 21, 'U': 22, 'V': 23, 'W': 24,
              'X':25, 'Y': 26, 'Z': 27}

def load_NIST(path):
    x,y = [],[]

    for dir_name in os.listdir(path):

        sub = '/'.join([path,dir_name])

        if os.path.isdir(sub):

            for img in os.listdir(sub):
                p = '/'.join([sub,img])

                img = imread(p,as_gray=True)
                img_r = resize(img,(28,28))
                x.append(img_r.flatten()) # flatten for other clfs
                y.append(LETTER_MAP[dir_name])

    x = np.array(x)
    y = np.array(y)

    x, y = shuffle(x, y)

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.20)

    return X_train,y_train,X_test,y_test


# Load OCR Data
x_train_NIST,y_train_NIST,x_test_NIST,y_test_NIST = load_NIST(DATA)


# # Perceptron Train
# pr_clf = Perceptron()
# results = cross_val_score(pr_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
# mean_accu = results.mean()
# print('Perceptron Training Accuracy: %.2f'%(mean_accu))

# # Logistic Regression Train
# lr_clf = LogisticRegression()
# results = cross_val_score(lr_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
# mean_accu = results.mean()
# print('Logistic Regression Training Accuracy: %.2f'%(mean_accu))

# # Gaussian NB Train
# gnb_clf = GaussianNB()
# results = cross_val_score(gnb_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
# mean_accu = results.mean()
# print('Gaussian NB Training Accuracy: %.2f'%(mean_accu))

# Decision Tree Train
dt_clf = DecisionTreeClassifier()
results = cross_val_score(dt_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
mean_accu = results.mean()
print('Decision Tree Training Accuracy: %.2f'%(mean_accu))

# Random Forest Train
rf_clf = RandomForestClassifier()
results = cross_val_score(rf_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
mean_accu = results.mean()
print('Random Forest Training Accuracy: %.2f'%(mean_accu))

# Neural Network Train
nn_clf = MLPClassifier()
results = cross_val_score(nn_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
mean_accu = results.mean()
print('Neural Network Training Accuracy: %.2f'%(mean_accu))

# Ensemble Train
estimators = [('dt',dt_clf),('rf',rf_clf),('nn',nn_clf),
              ('pr',pr_clf),('lr',lr_clf),('gnb',gnb_clf)]
en_clf = VotingClassifier(estimators,voting='hard')
results = cross_val_score(en_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
mean_accu = results.mean()
print('Ensemble Training Accuracy: %.2f'%(mean_accu))







LABELS = ['%d' % i for i in range(28)]

# # Perceptron Test
# pr_clf.fit(x_train_NIST,y_train_NIST)
# y_pred = pr_clf.predict(x_test_NIST)
# print("Perceptron Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
# print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

# # Logistic Regression Test
# lr_clf.fit(x_train_NIST,y_train_NIST)
# y_pred = lr_clf.predict(x_test_NIST)
# print("Logistic Regression Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
# print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

# # Gaussian NB Test
# gnb_clf.fit(x_train_NIST,y_train_NIST)
# y_pred = gnb_clf.predict(x_test_NIST)
# print("Gaussian NB Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
# print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

# Decision Tree Test
dt_clf.fit(x_train_NIST,y_train_NIST)
y_pred = dt_clf.predict(x_test_NIST)
print("Decsision Tree Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

# Random Forest Test
rf_clf.fit(x_train_NIST,y_train_NIST)
y_pred = rf_clf.predict(x_test_NIST)
print("Random Forest Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

# Neural Network Test
nn_clf.fit(x_train_NIST,y_train_NIST)
y_pred = nn_clf.predict(x_test_NIST)
print("Neural Network Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

# Ensemble Test
en_clf.fit(x_train_NIST,y_train_NIST)
y_pred = en_clf.predict(x_test_NIST)
print("Ensemble Test Accuracy: ", accuracy_score(y_test_NIST, y_pred))
print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))