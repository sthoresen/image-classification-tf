from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.misc

print(tf.__version__)

(images_train, labels_train), (images_test, labels_test) = keras.datasets.mnist.load_data()
im = np.concatenate((images_train, images_test))
lb = np.concatenate((labels_train, labels_test))
files = 0
os.makedirs('dataset')

for i in range(0,im.shape[0]):
    filename = "dataset/{:05d} is {:d}.png".format(i, lb[i])
    scipy.misc.imsave(filename, im[i])
    files += 1

print('wrote {:d}/{:d} images to /dataset.'.format(files, im.shape[0]))
