import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with zipfile.ZipFile("data.zip", "r") as file:
    file.extractall("data/")

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

path = os.getcwd()
parent = os.path.dirname(path)
os.chdir(parent)
current = os.getcwd()
sample = os.path.join(current, 'sample-images')
try:
    os.mkdir(sample)
except:
    pass

i = 0
for im in X_test[:31]:
    plt.figure(figsize=(2,2))
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.savefig(sample + '/' + str(i) + '.png', bbox_inches='tight', pad_inches=-0.1)
    i += 1
    plt.close()
  
print("Done!")
