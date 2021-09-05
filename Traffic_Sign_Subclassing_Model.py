# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:29:53 2021

@author: ahmed
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import random
import os
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split



train_classes_path = "E:/Software/professional practice projects/In progress/Train"
train_csv_path = "E:/Software/professional practice projects/In progress/Train.csv"

train_classes_df = pd.read_csv(train_csv_path)
print(train_classes_df.head())


# Exploring random image items
train_classes = os.listdir(train_classes_path)

counter = 0
plt.figure(figsize=(12, 8))
for class_name in train_classes:
    counter += 1
    
    random_class = random.choice(train_classes)
    random_class_path = os.path.join(train_classes_path, random_class)
    
    random_img = random.choice(os.listdir(random_class_path))
    random_img_path = os.path.join(random_class_path, random_img)
    
    img = cv2.imread(random_img_path)
    
    plt.subplot(2, 4, counter)
    plt.tight_layout()
    plt.imshow(img)
    plt.xlabel(img.shape[1])
    plt.ylabel(img.shape[0])
    plt.show()
    
    if counter == 8:
        break



dim1, dim2 = [], []

for class_name in train_classes:
    class_path = os.path.join(train_classes_path, class_name)
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        dim1.append(img.shape[0])
        dim2.append(img.shape[1])
    

dim1_mean = np.mean(dim1)
dim2_mean = np.mean(dim2)

print('dim1_mean ==> ', dim1_mean)
print('dim2_mean ==> ', dim2_mean)


img_width = int(dim1_mean)
img_height = int(dim2_mean)

print('img_width ==> ', img_width)
print('img_height ==> ', img_height)



def scale_image(img_path, width, height):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height))
    img = np.array(img)
    img = (img / 255.0)
    
    return img



images_dataset = []
labels = []
counter = -1

for class_name in train_classes:
    class_path = os.path.join(train_classes_path, class_name)
    counter += 1
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        img = scale_image(img_path, img_width, img_height)
        
        images_dataset.append(img)
        labels.append(counter)


images_dataset = np.array(images_dataset)
labels = np.array(labels)


# checking that images shapes are balanced
label_counts = pd.DataFrame(labels).value_counts()

# Splitting data into train and validation
x_train, x_val, y_train, y_val = train_test_split(images_dataset, labels, test_size=0.2, shuffle=True)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

num_classes = len(train_classes)


# Creating Model (Subclassing)
class TrafficSignModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(TrafficSignModel, self).__init__()
        
        self.conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_width, img_height, 3))
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv3 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
        
        self.btn1 = BatchNormalization()
        self.btn2 = BatchNormalization()
        self.btn3 = BatchNormalization()
        
        self.mxpool = MaxPooling2D(pool_size=(2, 2))
        self.dropout = Dropout(0.5)
        
        self.flat = Flatten()
        self.dense = Dense(128, activation='relu')
        self.classifier = Dense(num_classes, activation='softmax')
    
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.mxpool(x)
        x = self.btn1(x)
       
        
        x = self.conv2(x)
        x = self.mxpool(x)
        x = self.btn2(x)
        
        x = self.conv3(x)
        x = self.btn3(x)
        
        x = self.flat(x)
        x = self.dense(x)
        x = self.dropout(x)
        
        return self.classifier(x)


class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold
    
    
    def on_epoch_end(self, epoch, logs=None):
        ratio = logs['acc'] / logs['val_acc']
        print(" ratio = ", ratio)
        
        if ratio > self.threshold:
            print("Stop training because overfitting..")
            self.model.stop_training = True



model = TrafficSignModel(num_classes)


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val), verbose=1, 
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
                     CSVLogger('training.csv')])



plt.figure(figsize=(12, 8))
plt.plot(model.history.history['accuracy'], color='g')
plt.plot(model.history.history['val_accuracy'], color='b')
plt.title("Traffic_Sign_Model_Performance (Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()











