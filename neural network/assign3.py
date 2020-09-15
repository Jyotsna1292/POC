# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:59:14 2020

@author: USER
"""

# importing libraries
import numpy as np
import pandas as pd

import tensorflow as tf

tf.__version__

# importing dataset
concrete = pd.read_csv('C:/Users/USER/Downloads/concrete.csv')
x = startup.iloc[:, :-1].values
y = startup.iloc[:, -1].values

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# initializing the ANN
ann = tf.keras.models.Sequential()

# adding 1st hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# adding 2nd hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# adding the output layer
ann.add(tf.keras.layers.Dense(units=1 ))

# compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
X = np.asarray(x_train).astype(np.float32)
Y=  np.asarray(y_train).astype(np.float32)
# training the ANN model on training set
ann.fit(X,Y, batch_size=32, epochs=100)

# prediction
x_test=np.asarray(x_test).astype(np.float32)
y_test=np.asarray(y_test).astype(np.float32)
y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
