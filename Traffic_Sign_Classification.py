# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:50:21 2021

@author: Ahmed Fayed
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, f1_score, roc_auc_score, roc_curve
# from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten



train = pd.read_csv('Train.csv')
meta = pd.read_csv('Meta.csv')
test = pd.read_csv('Test.csv')



# images have different sizes so we're going to take the average of dimensions
dim1 = []
dim2 = []

train_path = 'E:/Software/Practise Projects/Traffic Sign Classification/Train/'
classes_num = os.listdir(train_path)


for num in range(len(classes_num)):
    class_dir = train_path + str(num)
    class_images = os.listdir(class_dir)
    
    for img_name in class_images:
        img_path = os.path.join(class_dir + '/', img_name)
        
        img = cv2.imread(img_path)
        
        dim1.append(img.shape[0])
        dim2.append(img.shape[1])
        

dim1_mean = np.mean(dim1)
dim2_mean = np.mean(dim2)
    

# Resizing images to the average of the dimensions
# Normalizing Images
Images_data = []
Labels = []
#norm_obj = StandardScaler(copy=True, with_mean=True, with_std=True)

for num in range(len(classes_num)):
    class_dir = train_path + str(num)
    class_images = os.listdir(class_dir)
    
    for img_name in class_images:
        img_path = os.path.join(class_dir + '/', img_name)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (50, 50))
        
        # Normalizing Image
        img = img / 255
        
        Images_data.append(img)
        Labels.append(num)
        
        
  
# number of images per class      
label_counts = pd.DataFrame(Labels).value_counts()

# spliiting data
Images_data = np.array(Images_data)
X_train, X_val, Y_train, Y_val = train_test_split(Images_data, Labels, test_size=0.2, shuffle=True)

X_train.shape[1:]

# making one hot encodding
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)


# Creating the Model Architecture
model = Sequential()

model.add(Conv2D(filters=128, kernel_size=(7, 7), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(43, activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_val, Y_val), verbose=2)


# plotting the model
Evaluation = pd.DataFrame(model.history.history)
Evaluation[['accuracy', 'val_accuracy']].plot()
Evaluation[['loss', 'val_loss']].plot()



# Reading and Resizing test data
test_path = 'E:/Software/Practise Projects/Traffic Sign Classification/Test/'
test_images = []

for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50))
    
    # Normalizing image
    img = img / 255.0
    
    test_images.append(img)
    

test_images = np.array(test_images)


# Labeling testing images

test_labels = test['ClassId']
test_labels = np.array(test_labels)

predictions = model.predict(test_images)
predictions_class = model.predict_classes(test_images)


# Evaluating model with testing data

accuracy_score = accuracy_score(test_labels, predictions_class)
confusion_matrix = confusion_matrix(test_labels, predictions_class)
mean_squared_error = mean_squared_error(test_labels, predictions_class)
f1_score = f1_score(test_labels, predictions_class, average='micro')














