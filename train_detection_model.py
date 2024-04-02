import pandas as pd
import mat73
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import confusion_matrix


class DetectionModel:
    def __init__(self, dataset_path):
        self.data = tf.keras.utils.image_dataset_from_directory(dataset_path, batch_size=32, image_size=(32,32))

    def prepare_data(self):
        #scaling data between 0 and 1
        data = self.data.map(lambda x,y: (x/255,y))
        train_size = int(len(data)*0.7)
        val_size = int(len(data)*0.2)
        test_size = int(len(data)) - train_size - val_size

        train_dataset = self.data.take(train_size)
        val_dataset = self.data.skip(train_size).take(val_size)
        test_dataset = self.data.skip(train_size+val_size).take(test_size)
        return train_dataset, val_dataset, test_dataset

    def model_building(self):
        model = Sequential()
        model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    
    def model_performance(self, model, test):
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()
        y_true = list()
        y_pred = list()
        for batch in test.as_numpy_iterator():
            X,Y = batch
            pred = model.predict(X)
            y_true.append(Y)
            y_pred.append(pred)
            pre.update_state(Y, pred)
            re.update_state(Y, pred)
            acc.update_state(Y, pred)

        cm = confusion_matrix(np.concatenate(y_true,axis=0), np.round(np.concatenate(y_pred,axis=0)))
    
        print("precision:", pre.result().numpy())
        print("recall:", re.result().numpy())
        print("accuracy:", acc.result().numpy())
        print("confusion matrix:", cm)

    def train(self,model_path):
        train, val, test = self.prepare_data()
        model = self.model_building()
        model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
        model.fit(train, epochs = 100, validation_data=val)
        model.save(model_path)
        self.model_performance(model,test)
        return model
    

    
if __name__ == "__main__":
    dataset_path = 'data_for_detection'
    model_path = "models/detection_model.h5"
    model_ = DetectionModel(dataset_path)
    trained_model = model_.train(model_path)
    
