from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import load_model
import os

import numpy as np
import load_photo as LOAD

x_train,y_train = LOAD.load("/home/alex/Documents/Robo_hand/photos")


x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)

Input = Input(shape=(100,100,1), dtype='float32', name='Input')
Conv1 = Conv2D(30, (3, 3),activation = "relu",name="conv1")(Input)
Pool1 = MaxPooling2D(pool_size=(2, 2),name = "pool")(Conv1)
Drop1 = Dropout(0.25,name="drop1")(Pool1)
Conv2 = Conv2D(30, (3, 3),activation = "relu",name="conv2")(Drop1)
Pool2 = MaxPooling2D(pool_size=(2, 2),name = "pool2")(Conv2)
Drop2 = Dropout(0.25,name="drop2")(Pool2)

Flat = Flatten(name = "Flat")(Drop2)
Dens1 = Dense(256,activation = "relu",name = "dens1")(Flat)
Drop3 = Dropout(0.25,name = "drop3")(Dens1)
OUT = Dense(len(y_train),activation = "softmax",name = "output")(Drop3)

model = Model(inputs=Input, outputs=OUT)
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_train/= 255
y_train = np_utils.to_categorical(y_train,len(y_train))

batch_size = 32
epochs = 200

model.summary()

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True)

# Записываем модель в файл
json_file = open("block_classifier_model.json", "w")
json_string = model.to_json()
json_file.write(json_string)
json_file.close()

model.save_weights("block_classifier_weights.h5")


print("Trained succsesfully!")


