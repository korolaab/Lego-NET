import keras
from keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
import PIL
from PIL import Image
import numpy as np

Input_Main = Input(shape=(480,640,1), dtype='float32', name='Input_Main')
Conv_I = Conv2D(128, (10, 10),activation = "relu",name="conv_I")(Input_Main)
#Input = Input(shape=(100,100,1),dtype='float32', name='Input')(Conv_I)

Conv1 = Conv2D(30, (3, 3),activation = "relu",name="conv1")(Conv_I)
Pool1 = MaxPooling2D(pool_size=(2, 2),name = "pool")(Conv1)
Drop1 = Dropout(0.25,name="drop1")(Pool1)

Conv2 = Conv2D(30, (3, 3),activation = "relu",name="conv2")(Drop1)
Pool2 = MaxPooling2D(pool_size=(2, 2),name = "pool2")(Conv2)
Drop2 = Dropout(0.25,name="drop2")(Pool2)

PoolOut = MaxPooling2D(pool_size=(2, 2),name = "poolout")(Drop2)

Flat = Flatten(name = "Flat")(Drop2)
Dens1 = Dense(256,activation = "relu",name = "dens1")(Flat)
Drop3 = Dropout(0.25,name = "drop3")(Dens1)
OUT = Dense(7,activation = "softmax",name = "output")(Drop3)

model = Model(inputs=Input_Main, outputs=[OUT,PoolOut])
model.load_weights('block_classifier_weights.h5',by_name=True)
model.compile(loss='categorical_crossentropy',
              metrics={'output': 'accuracy'})

model.summary()

im  = Image.open("3.JPG").convert("L")
pix = np.array(im)
pix = np.expand_dims(pix , axis=3)
pix = np.expand_dims(pix,axis = 0)
print(pix.shape)
answer , x = model.predict(pix)

print(answer)
