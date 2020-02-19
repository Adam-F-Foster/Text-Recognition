# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:22:06 2019

@author: Adam
"""

import numpy as np
from sklearn import model_selection as ms
from emnist import extract_training_samples
from tensorflow import keras
images, labels = extract_training_samples("letters")
labels.setflags(write = 1)
for i in range(len(labels)):
    labels[i] = labels[i] - 1
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
        print("formatting failed")     


trainX, testX, trainY, testY = ms.train_test_split(images, labels, test_size = 0.2)
#trainX = [np.reshape(formatImage(trainX[i]), (784,)) for i in range(len(trainX))]
#testX = [np.reshape(formatImage(testX[i]), (784,)) for i in range(len(testX))]
trainX = np.array(trainX)
testX = np.array(testX)
trainY = np.array(trainY)
testY = np.array(testY)

trainX = trainX.reshape(len(trainX), 28, 28, 1)
testX = testX.reshape(len(testX), 28, 28, 1)
print(len(testX))
trainX = trainX / 255
testX = testX / 255
trainY = keras.utils.to_categorical(trainY, 26)
testY = keras.utils.to_categorical(testY, 26)


model = keras.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size = 2, activation = "relu", input_shape = (28,28, 1)))
model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = "relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation = "relu"))
model.add(keras.layers.Dense(26, activation = "softmax"))


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])    
model.fit(trainX, trainY, epochs = 1)
print("\n\n")
loss, acc = model.evaluate(testX, testY)
print("Test Accuracy:", acc)

class cnn:
    def __init__(self, acc, model):
        self.acc = acc
        self.model = model
result = cnn(acc, model)
print("done")
