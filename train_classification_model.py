import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras``
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(dataset_path):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('jpg'):
            img = cv2.imread(os.path.join(dataset_path,filename),cv2.IMREAD_GRAYSCALE)
            img =  cv2.resize(img, [32,32])
            img = np.reshape(img, [32,32,1])
            img_ = img.astype('float32') / 255.0
            label = int(filename.split('_')[1][:-4])
            if label ==  10:
                label = 0
            images.append(img_)
            labels.append(label)
    return np.array(images), np.array(labels)

## splitting dataset 
images,labels = load_data('data_for_detection/digits')
X_train, X_test, Y_train,Y_test = train_test_split(images,labels,test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42) 

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

n_classes = 10

Y_train = to_categorical(Y_train, n_classes)
Y_test = to_categorical(Y_test, n_classes)
Y_val = to_categorical(Y_val, n_classes)


def model_building(n_classes):
    model = Sequential()
    model.add(Conv2D(128, (3,3), activation='relu', input_shape = (32,32,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model

model = model_building(n_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data = (X_val, Y_val))
model.save('models/classification_model.h5')
loss,accuracy = model.evaluate(X_test, Y_test)

print(loss, accuracy)
