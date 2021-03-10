# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:50:21 2021

@author: Ahmed Fayed
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical




train = pd.read_csv('Train.csv')
meta = pd.read_csv('Meta.csv')
test = pd.read_csv('Test.csv')







