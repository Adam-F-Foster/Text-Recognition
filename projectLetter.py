# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:08:24 2019
@author: adamf
"""
import numpy as np
import cv2
from sklearn import dummy, svm, metrics, tree, model_selection as ms
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
from emnist import extract_training_samples
import cnnKeras

alphabet = 'abcdefghijklmnopqrstuvwxyz'
alphabet = alphabet.upper()
def formatImage(img):
    image = []
    for elem in img:
        image.append(elem)
    try:
        image = np.reshape(image, (28, 28))
        flip = np.fliplr(image)
        rot = np.rot90(flip)
        return rot
    except:
        print("oopsie")
def knn(trainingX, trainingY, example):
    sim = 0
    index = 0
    for i in range(len(trainingX)):
        same = countSimilarity(trainingX[i], example[0])
        if same > sim:
            sim = same
            index = i   
    return alphabet[trainingY[index] - 1]
def countSimilarity(lst, lst1):
    lst = list(lst)
    lst1 = list(lst1)
    count = 0
    for i in range(len(lst)):
        if lst[i] == lst1[i] and lst[i] > 150:
            count += 1
    return count
file = input("Please enter a file name - xxxxx.png: ")
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
ret, new_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
dilated = cv2.dilate(new_img, kernel, iterations=5)  # dilate , more the iteration more the dilation
image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

name = ""
digits = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 40 and h < 40:
        continue
    digits.append(dilated[y+2:y-2+h, x+2:x-2+w])

images, labels = extract_training_samples("letters")
#i would avoid using more than 8000 samples for speed's sake
images = images[:4000]
labels = labels[:4000]

trainX, testX, trainY, testY = ms.train_test_split(images, labels, test_size = 0.2)
trainX = [np.reshape(formatImage(trainX[i]), (784,)) for i in range(len(trainX))]
testX = [np.reshape(formatImage(testX[i]), (784,)) for i in range(len(testX))]
trainX = np.array(trainX)
testX = np.array(testX)
trainY = np.array(trainY)
testY = np.array(testY)

clf = svm.SVC(kernel = "linear")
clf.fit(trainX, trainY)

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(trainX, trainY)

dum = dummy.DummyClassifier(strategy = "stratified")
dum.fit(trainX, trainY)

cnn = cnnKeras.result.model

skKnn = KNeighborsClassifier(n_neighbors=3)
skKnn.fit(trainX, trainY)
print("Analyzing Character")
for digit in digits:

    digit = cv2.resize(digit, (28, 28), interpolation = cv2.INTER_AREA)
    plt.figure()
    plt.imshow(digit, cmap = plt.cm.gray)
    array = np.reshape(digit, (1, 784))
    
    start = time.time()
    predY = clf.predict(array)
    print("Linear:", alphabet[predY[0] - 1], "Time:", time.time() - start)
    
    start = time.time()
    predY = skKnn.predict(array)
    print("sklearn KNN:", alphabet[predY[0] - 1], "Time:", time.time() - start)
    
    start = time.time()
    treeY = clf_tree.predict(array)
    print("Decision Tree:", alphabet[treeY[0] - 1], "Time:", time.time() - start)
    
    start = time.time()
    dumY = dum.predict(array)
    print("Dummy:", alphabet[dumY[0] - 1], "Time:", time.time() - start)
    
    start = time.time()
    print("Self Implement KNN:", knn(trainX, trainY, array), "Time:", time.time() - start)
    
    array = array.reshape(1, 28, 28, 1)
    start = time.time()
    print("Self Implement CNN:", alphabet[cnn.predict_classes(array)[0]], "Time:", time.time() - start)
    print("\n")

predY = clf.predict(testX)
svmacc = metrics.accuracy_score(testY, predY)
print("SVM SVC:", svmacc)

predY = skKnn.predict(testX)
skacc = metrics.accuracy_score(testY, predY)
print("sklearn KNN", skacc)
predY = clf_tree.predict(testX)
dtacc = metrics.accuracy_score(testY, predY)
print("Decision Tree:", dtacc)

predY = dum.predict(testX)
dumacc = metrics.accuracy_score(testY, predY)
print("Dummy Classifier:", dumacc)


knns = []
actual = [alphabet[testY[i]-1] for i in range(20)]#testY[:20]
rows = testX[:20]


for row in rows:
    row = np.reshape(row, (1, 784))
    knns.append(knn(trainX, trainY, row))
knnacc = metrics.accuracy_score(actual, knns)
print("Self Implemented KNN:", knnacc)

cnn = cnnKeras.result.acc
print("Self Implemented CNN:", cnn)
accuracies = [svmacc, dtacc, dumacc, knnacc, cnn, skacc]
labels = ["SVM", "Decision Tree", "Dummy Classifier", "KNN", "CNN", "SKKNN"]

plt.figure()
plt.bar(x = labels, height = accuracies)
print("main done")




