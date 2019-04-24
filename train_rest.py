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
import matplotlib.pyplot as plt

MODEL_PATH = './cnn.h5'
DATA = '../data'
LETTER_MAP = {'A' : 0, 'B' : 1, 'C': 2, 'D': 3,
              'E': 4, 'F':5,'G':6, 'H':7, 'I':8,
              'i': 9, 'J':10, 'K': 11, 'L': 12,
              'l':13, 'M':14, 'N':15, 'O': 16,
              'P': 17, 'Q' : 18, 'R': 19, 'S': 20,
              'T': 21, 'U': 22, 'V': 23, 'W': 24,
              'X':25, 'Y': 26, 'Z': 27}

DIM = 28 # pixels
BARG_FIGURE_SIZE = (6,5)

def plot_results(test_results,train_results):
    
    ind = np.arange(4) 
    width = 0.35
    
    clf_test_scores, clf_train_scores, clf_names = [], [], []

    for clf,score in train_results:
        clf_train_scores.append(score)

    for clf,score in test_results:
        clf_test_scores.append(score)
        clf_names.append(clf)

    plt.figure(figsize=BARG_FIGURE_SIZE)
    plt.bar(ind, clf_test_scores,width,label='Test', align='center', alpha=0.5)
    plt.bar(ind + width, clf_train_scores,width,label='Train', align='center', alpha=0.5)
    plt.legend(loc='best')
    plt.xticks(ind + width / 2, clf_names)
    plt.ylabel('accuracy')
    plt.xlabel('classifier')
    plt.title('Classifier Accuracies')
    
    plt.savefig('ClassifiersAccuracy.png')

    plt.cla()

def load_NIST(path):
    x,y = [],[]
    for dir_name in os.listdir(path):
        sub = '/'.join([path,dir_name])
        counter = 0
        if os.path.isdir(sub):
            for img in os.listdir(sub):
                p = '/'.join([sub,img])
                img = imread(p,as_gray=True)
                img_r = resize(img,(DIM,DIM))
                x.append(img_r.flatten()) # flatten for other clfs
                y.append(LETTER_MAP[dir_name])
                counter += 1
                if counter >= 1000:
                    break
    x = np.array(x)
    y = np.array(y)

    x, y = shuffle(x, y)

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.20)

    return X_train,y_train,X_test,y_test

if __name__ == '__main__':
    # Load OCR Data
    x_train_NIST,y_train_NIST,x_test_NIST,y_test_NIST = load_NIST(DATA)

    # Decision Tree Train
    dt_clf = DecisionTreeClassifier()
    results = cross_val_score(dt_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
    dt_train_accu = results.mean()
    print('Decision Tree Training Accuracy: %.2f'%(dt_train_accu))

    # Random Forest Train
    rf_clf = RandomForestClassifier()
    results = cross_val_score(rf_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
    rf_train_accu = results.mean()
    print('Random Forest Training Accuracy: %.2f'%(rf_train_accu))

    # Neural Network Train
    nn_clf = MLPClassifier()
    results = cross_val_score(nn_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
    nn_train_accu = results.mean()
    print('Neural Network Training Accuracy: %.2f'%(nn_train_accu))

    # Ensemble Train
    estimators = [('dt',dt_clf),('rf',rf_clf),('nn',nn_clf)]
    en_clf = VotingClassifier(estimators,voting='hard')
    results = cross_val_score(en_clf,x_train_NIST,y_train_NIST,cv=5,scoring=('accuracy'))
    en_train_accu = results.mean()
    print('Ensemble Training Accuracy: %.2f'%(en_train_accu))


    LABELS = ['%d' % i for i in range(28)]


    # Decision Tree Test
    dt_clf.fit(x_train_NIST,y_train_NIST)
    y_pred = dt_clf.predict(x_test_NIST)
    dt_test_accu = accuracy_score(y_test_NIST, y_pred)
    print("Decsision Tree Test Accuracy:",dt_test_accu)
    print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

    # Random Forest Test
    rf_clf.fit(x_train_NIST,y_train_NIST)
    y_pred = rf_clf.predict(x_test_NIST)
    rf_test_accu = accuracy_score(y_test_NIST, y_pred)
    print("Random Forest Test Accuracy:",rf_test_accu )
    print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

    # Neural Network Test
    nn_clf.fit(x_train_NIST,y_train_NIST)
    y_pred = nn_clf.predict(x_test_NIST)
    nn_test_accu = accuracy_score(y_test_NIST, y_pred)
    print("Neural Network Test Accuracy:",nn_test_accu)
    print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

    # Ensemble Test
    en_clf.fit(x_train_NIST,y_train_NIST)
    y_pred = en_clf.predict(x_test_NIST)
    en_test_accu = accuracy_score(y_test_NIST, y_pred)
    print("Ensemble Test Accuracy:",en_test_accu)
    print(classification_report(y_test_NIST, y_pred, target_names=LABELS, digits=3))

    train_results = [('DT',dt_train_accu),('RF',rf_train_accu),('NN',nn_train_accu),('EN',en_train_accu)]

    test_results = [('DT',dt_test_accu),('RF',rf_test_accu),('NN',nn_test_accu),('EN',en_test_accu)]

    plot_results(train_results,test_results)