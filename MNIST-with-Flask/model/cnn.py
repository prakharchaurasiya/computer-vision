import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

with zipfile.ZipFile('data.zip', 'r') as file:
    file.extractall('data/')

train = pd.read_csv('data/mnist_train_small.csv')
test = pd.read_csv('data/mnist_test.csv')

# split X, y

X_train = train.iloc[:,1:]
y_train = pd.get_dummies(train.iloc[:,0]) # one hot encoding
X_test = test.iloc[:,1:]
y_test = pd.get_dummies(test.iloc[:,0]) # one hot encoding

# to numpy

X_train = (X_train.values).reshape(X_train.shape[0], 28, 28) # reshape to 3d
y_train = y_train.values
X_test = (X_test.values).reshape(X_test.shape[0], 28, 28) # reshape to 3d
y_test = y_test.values

#----------Model----------

model = Sequential()
model.add(Conv2D(16, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test)

model.save('MNIST.h5')
