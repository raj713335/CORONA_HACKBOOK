import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.getcwd()

print(dir_path)


covid19_patients=dir_path+'/corona_check/train/COVID19'
normal_patients = dir_path+'/corona_check/train/Normal'

print(covid19_patients)

print('total training covid19 patients images:', len(os.listdir(covid19_patients)))
print('total training noraml patients images:', len(os.listdir(normal_patients)))






one_files = os.listdir(covid19_patients)
print(one_files[:])

two_files = os.listdir(normal_patients)
print(two_files[:])



import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_one = [os.path.join(covid19_patients, fname)
             for fname in one_files[pic_index - 2:pic_index]]
next_two = [os.path.join(normal_patients, fname)
              for fname in two_files[pic_index - 2:pic_index]]


for i, img_path in enumerate(next_one + next_two):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    if i>10:
        break


import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


TRAINING_DIR=dir_path+'/corona_check/train'
training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,  # The magnitude of the vertical shift of the picture when the data is boosted
    shear_range = 0.2,         # Set the shear strength
    horizontal_flip = True,
    zoom_range=0.2,
    vertical_flip=True,
    fill_mode ='nearest'
)



VALIDATION_DIR = dir_path+"/corona_check/test"

image_size = 150
batch_size = 50
nb_classes = 2



validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    (image_size, image_size),  # Integer tuple (height, width), default: (256, 256). All images will be resized
    batch_size=batch_size,  # The size of the batch of data (default 32)
    class_mode='categorical'
)
print(train_generator)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    (image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)


train_steps = train_generator.samples//batch_size # "//" means integer division
test_steps = validation_generator.samples//batch_size

# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('acc')>0.90):
#       print("\nReached 90% accuracy so cancelling training!")
#       self.model.stop_training = True
#
#
#
#
#
# callbacks = myCallback()


# ACCURACY_THRESHOLD = 0.95

# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#       if(logs.get('acc') > ACCURACY_THRESHOLD):
#           print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
#           self.model.stop_training = True
#
# callbacks = myCallback()




# FEEDBACK PYUISH USE BATCH NORMALIZATION, REDURE DROUPOUT to 1% and use DIALATION AND EROSION WITH CUT-OFFS
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




model.summary()



from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps,
                              epochs=20,
                              validation_data=validation_generator,
                              verbose=1,
                              validation_steps=8
                              )

model.save("Coronavirus.h5")

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

print("END")