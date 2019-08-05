from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import shutil
import os

print(tf.__version__)

# Import the MNIST dataset from keras
(images_train, labels_train), (images_test, labels_test) = keras.datasets.mnist.load_data()

# Explore the data
print('training set shape', images_train.shape)
print('labels train shape', labels_train.shape)
print('image shape', images_train[0].shape)
print('labels test shape', labels_test.shape)
print('testing set shape', images_test.shape)

# Preprocess (Normalize images)
images_train = images_train / 255;
images_test = images_test / 255;

# Design the model
model = tf.keras.Sequential()
RELU_UNITS = 128;
model.add(layers.Flatten(input_shape=(28, 28)))
# Adds a densely-connected layer with RELU_UNITS units to the model:
model.add(layers.Dense(RELU_UNITS, activation='relu'))
# Add another:
model.add(layers.Dense(RELU_UNITS, activation='relu'))
# Add a softmax layer with OUT_UNITS output units:
model.add(layers.Dense(10, activation='softmax'))


# Configure a model for categorical classification.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the net
history = model.fit(images_train, labels_train, epochs=15, batch_size=512,
          validation_data=(images_test, labels_test))

# Plot test accuracy vs validation accuracy, to visualize overfitting
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'], 'b', label="Training acc")
plt.plot(epochs, history.history['val_acc'], 'r', label="Validation acc")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy plots')
plt.legend()
plt.show()

# Create a directory 'fails' to visualize all misclassifications
predictions = model.predict(images_test)
misclassified = np.zeros(10)
shutil.rmtree('fails', ignore_errors=True)
os.makedirs('fails')

# Plot a bar-graph displaying frequency of error per symbol
for i in range(0,predictions.shape[0]-1):
    if np.argmax(predictions[i]) != labels_test[i]:
        #print(misclassified.shape)
        #print(labels_test.shape)
        #exit()
        misclassified[labels_test[i]] += 1
        filename = "fails/actual:{:d} guessed:{:d}.png".format(labels_test[i], np.argmax(predictions[i]))
        scipy.misc.imsave(filename, images_test[i])

plt.bar(np.arange(10), misclassified)
plt.xlabel('Symbols 0-9')
plt.ylabel('Misclassifications')
plt.title('Misclassifications per symbol')
plt.show()
