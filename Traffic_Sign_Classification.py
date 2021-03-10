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
# from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical




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
X_train, X_test, Y_train, Y_test = train_test_split(Images_data, Labels, test_size=0.2, shuffle=True)

# making one hot encodding
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)









