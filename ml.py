# import the necessary packages
# code inspire from https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2, l1

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import sys
import config
import glob
import cv2

# initializiation of the initial learning rate, the number of epochs ,
# and  the batch size
initial_learning_rate = 1e-3  # learning rate
Epochs = 10  # number of times that it is given to the model to be learned
batch = 32  # number of training example that will be processed just before it updates the weight

# Loop through the two file aircraft and Not_aircraft
# add each of them,which has been resized and in bgr, to a list
labels = {'Aircraft': 0, 'Not_aircraft': 1}
tr_img = []
tr_labs = []
for directory_path in glob.glob("/Users/kevintrang/Downloads/test/dataset/*"):
    label = directory_path.split("\\")[-1]
    # print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        tr_img.append(img)
        tr_labs.append(labels[label.split("/")[6]])

# put into array
data = np.array(tr_img)
labels = np.array(tr_labs)
# apply one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
(entX, evalX, entY, evalY) = train_test_split(data, labels,
                                              test_size=0.20, stratify=labels, random_state=42)
# increase the number of images
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.25,
    height_shift_range=0.10,
    shear_range=0.20,
    horizontal_flip=True,
    fill_mode="nearest")

netw = MobileNetV2(weights="imagenet", include_top=False,
                   input_tensor=Input(shape=(64, 64, 3)))  # taillee image et rgb

bas = netw.output  # utiliser la premiere dans la deuxieme
bas = AveragePooling2D(pool_size=(1, 1))(bas)  # avegerage pooling
bas = Flatten(name="flatten")(bas)  # flatten matrices to put it into a vector for later use
bas = Dense(128, activation="relu")(bas)  # number of neuron
bas = Dropout(0.5)(bas)  # avoid overlearning, between 0.4 et 0.6
bas = Dense(64, activation="relu")(bas)  # number of neuron
bas = Dropout(0.6)(bas)   # avoid overlearning, between 0.4 et 0.6
bas = Dense(2, activation="softmax")(bas)  # don't change the number of label

model = Model(inputs=netw.input, outputs=bas)

# loop over the layer to freeze it and to not update the data
for l in netw.layers:
    l.trainable = False
# divising by 255 to have a value near 0 (more stable)
entX = entX / 255
evalX = evalX / 255
# compile our model
# using optimizer to optimize the output
print("Compiling")  # a modif
opt = Adam(lr=initial_learning_rate)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

#parameters to train the network
H = model.fit(
    aug.flow(entX, entY, batch_size=batch),
    steps_per_epoch=len(entX) // batch,
    validation_data=(evalX, evalY),
    validation_steps=len(evalX) // batch,
    epochs=Epochs)

# Start the training
predIdxs = model.predict(evalX, batch_size=batch)

predIdxs = np.argmax(predIdxs, axis=1)
#Save model
print("save model")
model.save(config.modpath, save_format="h5")
#Save label
print("save label")
f = open(config.labpath, "wb")
f.write(pickle.dumps(lb))
f.close()
