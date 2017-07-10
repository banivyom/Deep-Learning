from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
import keras

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))


import numpy as np
np.random.seed(123)
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0],28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 
model = Sequential()
 
model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1,validation_data=(X_test, Y_test))
 
y = model.predict(X_test)
y_pred = np.zeros(len(X_test))
for i in range(len(y)):
    for j in range(len(y[i])):
        if y[i][j]>0.5:
            y_pred[i] = j
loss = sum(np.abs(y_test - y_pred))/float(len(y_pred))
y_pred = y_pred.astype(int)

score = model.evaluate(X_test, Y_test, verbose=0)

model.summary()
#99.30% accuracy on test set
