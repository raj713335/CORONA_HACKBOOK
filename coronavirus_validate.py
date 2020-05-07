# import the necessary packages
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#tf.logging.set_verbosity(tf.logging.ERROR)

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import datetime
import keras.models
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from skimage import transform



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])






model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.load_weights('Coronavirus.h5')

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (150, 150, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

try:
    image = load('OUTPUT/1.jpg')
except:
    image = load('OUTPUT/1.jpeg')


arry_predict=(model.predict(image))

print(arry_predict)

max=max(arry_predict[0])
index=list(arry_predict[0]).index(max)

print(max)
print(index)

print(arry_predict,"ddhd")

if max==arry_predict[0][0]:
    print("PATIENT TESTED  POSITIVE FOR COVID19")
else:
    print("PATIENT TESTED  NEGATIVE FOR COVID19")

exit(0)
