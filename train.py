import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.utils import np_utils 
from keras.models import load_model
import os
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
import load_photo as LOAD




def data():
    num_classes = 7
    x_train,y_train = LOAD.load("dataset")   #load photos and answers 
    x_train = x_train.astype('float32')
    x_train/= 255
    y_train = np_utils.to_categorical(y_train,num_classes)
    return x_train,y_train





#x_train = np.expand_dims(x_train, axis=3)
#print(x_train.shape)





###============================================================### model
def create_model(x_train,y_train):
    batch_size = 32
    epochs = 200
    num_classes = 7
    Input_main = Input(shape=(100,100,3), dtype='float32', name='Input_main')
    Conv1 = Conv2D(32, (3, 3),activation = "relu",name="conv1")(Input_main)
    Pool1 = MaxPooling2D(pool_size=(2, 2),name = "pool")(Conv1)
    Drop1 = Dropout({{uniform(0, 1)}},name="drop1")(Pool1)

    Conv2 = Conv2D(64, (3, 3),activation = "relu",name="conv2")(Drop1)
    Pool2 = MaxPooling2D(pool_size=(2, 2),name = "pool2")(Conv2)
    Drop2 = Dropout({{uniform(0, 1)}},name="drop2")(Pool2)

    Conv3 = Conv2D(128, (3, 3),activation = "relu",name="conv3")(Drop2)
    Pool3 = MaxPooling2D(pool_size=(2, 2),name = "pool3")(Conv3)
    Drop5 = Dropout({{uniform(0, 1)}},name="drop5")(Pool3)

    Flat = Flatten(name = "Flat")(Drop5)

    Dens1 = Dense(512,activation = "relu",name = "dens1")(Flat)
    Drop3 = Dropout({{uniform(0, 1)}},name = "drop3")(Dens1)

    Dens2 = Dense(512,activation = "relu",name = "dens2")(Drop3)
    Drop4 = Dropout({{uniform(0, 1)}},name = "drop4")(Dens2)

    OUT = Dense(num_classes,activation = "softmax",name = "output")(Drop4)

    model = Model(inputs=Input_main, outputs=OUT)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    

    model.summary()     
    
    model.fit(x_train, y_train,             ### training
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True)
    score = model.evaluate(x_train, y_train, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
###===================================================================### end model

best_run,best_model = optim.minimize(model=create_model,
                          data=data,
                          algo=tpe.suggest,
                          max_evals=5,
                          trials=Trials())

print("Trained succsesfully!")
print("Saving model to JSON")
json_file = open("block_classifier_model.json", "w") ### exporting model to json
json_string = best_model.to_json()
json_file.write(json_string)
json_file.close()
print("Succsess") 

print("Saving weights...")                                                  
best_model.save_weights("block_classifier_weights.h5")   ### saving weghts
print("Succsess")   





