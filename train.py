import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.utils import np_utils 
from keras.models import load_model
import os
import load_photo as LOAD



batch_size = 32
epochs = 200
num_classes = 7

x_train,y_train = LOAD.load("dataset")   #load photos and answers 


x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)

x_train = x_train.astype('float32')
x_train/= 255
y_train = np_utils.to_categorical(y_train,num_classes)




###============================================================### model

Input = Input(shape=(100,100,1), dtype='float32', name='Input')
Conv1 = Conv2D(32, (3, 3),activation = "relu",name="conv1")(Input)
Pool1 = MaxPooling2D(pool_size=(2, 2),name = "pool")(Conv1)
Drop1 = Dropout(0.25,name="drop1")(Pool1)

Conv2 = Conv2D(64, (3, 3),activation = "relu",name="conv2")(Drop1)
Pool2 = MaxPooling2D(pool_size=(2, 2),name = "pool2")(Conv2)
Drop2 = Dropout(0.25,name="drop2")(Pool2)


Flat = Flatten(name = "Flat")(Drop2)

Dens1 = Dense(512,activation = "relu",name = "dens1")(Flat)
Drop3 = Dropout(0.25,name = "drop3")(Dens1)

Dens2 = Dense(512,activation = "relu",name = "dens2")(Drop3)
Drop4 = Dropout(0.25,name = "drop4")(Dens2)

OUT = Dense(num_classes,activation = "softmax",name = "output")(Drop4)

model = Model(inputs=Input, outputs=OUT)
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

json_file = open("block_classifier_model.json", "w") ### exporting model to json
json_string = model.to_json()
json_file.write(json_string)
json_file.close()

###===================================================================### end model

model.summary()     

model.fit(x_train, y_train,             ### training
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True)

print("Trained succsesfully!")
print("Saving weights...")                                                  
model.save_weights("block_classifier_weights.h5")   ### saving weghts
print("Succsess")   





