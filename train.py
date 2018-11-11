import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.utils import np_utils 
from keras.models import load_model, model_from_json
#
import os
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
import load_photo as LOAD
import sys
from keras.utils import plot_model
import numpy
import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no debugging info TF

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
# exit()
from random import *
import cv2


# 




def data():
    num_classes = len(LOAD.arr_name)
    x_train,y_train = LOAD.load("dataset")   #load photos and answers 
    x_train = x_train.astype('float32')
    #
    #print(y_train)    
    y_train = np_utils.to_categorical(y_train,num_classes)


    x_test,y_test = LOAD.load("test_dataset")
    x_test = x_test.astype('float32')
    
    #print(y_train)    
    y_test = np_utils.to_categorical(y_test,num_classes)
    # print(y_train)
    # exit()
    datagen=ImageDataGenerator(
                        rescale=1./255,
                        brightness_range=(0.5,1),
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        channel_shift_range=0.2,
                        rotation_range=20,
                        zoom_range=0.2)
    datagen2=ImageDataGenerator(rescale=1./255,
                        width_shift_range=0.2,
                        height_shift_range=0.1,
                        channel_shift_range=0.1,
                        rotation_range=20,
                        zoom_range=0.1)
    return datagen,datagen2, x_train,y_train,x_test,y_test





#x_train = np.expand_dims(x_train, axis=3)
#print(x_train.shape)



from keras.optimizers import SGD
def Inception(datagen,valigen,x_train,y_train,x_test,y_test):
    numpy.random.seed(1)

    
    num_classes = len(LOAD.arr_name)
    Input_main = Input(shape=(100,100,3), dtype='float32', name='Input_main')
    convolution_matrix1 = 7
    pooling_matrix1 = 8

    # con=conditional({{choice([1,2])}})
    # if(con == 1):
    conv1 = Conv2D(32, (convolution_matrix1,convolution_matrix1),name='conv1',padding='same', activation='relu')(Input_main)
    pool1 = MaxPooling2D((pooling_matrix1,pooling_matrix1),padding='same',name='pool1')(conv1)
    drop1 = Dropout(0.25,name='drop1')(pool1)
    flat = Flatten(name='flat')(drop1)
   


    Dens1 = Dense(16,activation = 'relu',name = "dens1")(flat)
    Drop_last = Dropout(0.25,name = "drop_last")(Dens1)
    epochs = 10


    #earlystopper =keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,verbose=1)
  

    OUT = Dense(num_classes,activation = "softmax",name = "output")(Drop_last)

    model = Model(inputs=Input_main, outputs=OUT)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    

    model.summary()     
    
    model.fit(x_train, y_train,             ### training
              batch_size=32,
              epochs=epochs,
              verbose = 0,
              validation_split=0.1,
              shuffle=True)
    
    batch_size =32
    # model.fit_generator(
    #     datagen.flow(x_train, y_train,batch_size=32),
    #     steps_per_epoch=x_train.shape[0]// batch_size,
    #     epochs = epochs,verbose=0)
    score = model.evaluate(x_test, y_test, verbose=1)



# combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)  
    
    # model.fit(x_train, y_train,             ### training
    #           batch_size=32,
    #           epochs=10000,
    #           verbose = 1,
    #           validation_split=0.1,
    #           shuffle=True,
    #           callbacks = [earlystopper])
    print("Trained succsesfully!")
    print("Saving model to JSON")
    json_file = open("block_classifier_model.json", "w") ### exporting model to json
    json_string = model.to_json()
    json_file.write(json_string)
    json_file.close()
    print("Succsess")
    #print("Saving model to PNG")
   # plot_model(model, to_file='model.png')    
    #print("Succsess") 

    print("Saving weights...")                                                  
    model.save_weights("block_classifier_weights.h5")   ### saving weghts
    print("Succsess")   

    return 0
#x_train,y_train = data()
#a,b,c,d,e,f =data()
#Inception(a,b,c,d,e,f)
###============================================================### model
def train_model(datagen,valigen,x_train,y_train,x_test,y_test):
    numpy.random.seed(1)
    # generator(X_data=0,y_data=0,batch_size = 0)

    num_classes = len(LOAD.arr_name)
    Input_main = Input(shape=(100,100,3), dtype='float32', name='Input_main')
    convolution_matrix1 = {{choice([7,10,13,15,30,50])}}
    pooling_matrix1 = {{choice([2,4,8,12])}}

    # con=conditional({{choice([1,2])}})
    # if(con == 1):
    conv1 = Conv2D({{choice([32,64])}}, (convolution_matrix1,convolution_matrix1),name='conv1',padding='same', activation='relu')(Input_main)
    pool1 = MaxPooling2D((pooling_matrix1,pooling_matrix1),padding='same',name='pool1')(conv1)
    drop1 = Dropout(0.25,name='drop1')(pool1)
    flat = Flatten(name='flat')(drop1)


    Dens1 = Dense({{choice([32,64])}},activation = 'relu',name = "dens1")(flat)
    Drop_last = Dropout(0.25,name = "drop_last")(Dens1)
    epochs = 10


    #earlystopper =keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,verbose=1)
  

    OUT = Dense(num_classes,activation = "softmax",name = "output")(Drop_last)

    model = Model(inputs=Input_main, outputs=OUT)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    

    #model.summary()     
    
    # #model.fit(x_train, y_train,             ### training
    #           batch_size=32,
    #           epochs=epochs,
    #           verbose = 0,
    #           validation_split=0.1,
    #           shuffle=True)
    
    batch_size =32
    model.fit_generator(
        datagen.flow(x_train, y_train,batch_size=32),
        steps_per_epoch=x_train.shape[0]// batch_size,
        epochs = 1,verbose=0)
    score = model.evaluate(x_test,y_test,verbose=1)
    print(score)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
###===================================================================### end model
def main():
    best_run,best_model = optim.minimize(model=train_model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=100,
                              trials=Trials())
    datagen,valigen,x_train,y_train,x_test,y_test = data()

    print("Trained succsesfully!")
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_train,y_train))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.summary()
    print("Saving model to JSON")
    json_file = open("block_classifier_model_t.json", "w") ### exporting model to json
    json_string = best_model.to_json()
    json_file.write(json_string)
    json_file.close()
    print("Succsess") 

    print("Saving weights...")                                                  
    best_model.save_weights("block_classifier_weights_t.h5")   ### saving weghts
    print("Succsess")
    print("Starting training")
    os()
    return 0


def training():
    batch_size = 32
    epochs = 200
    json_file = open("block_classifier_model_t.json", "r")
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('block_classifier_weights_t.h5')
    #stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    earlystopper=keras.callbacks.EarlyStopping(monitor='val_loss', patience=100,verbose=1)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    d,v,x_train, y_train,x_test,y_test = data()

    batch_size =32

    model.fit_generator(
         d.flow(x_train,y_train,batch_size),
         validation_data=v.flow(x_test,y_test,batch_size),
         steps_per_epoch=2598 // batch_size,
         validation_steps=500//batch_size,
         epochs=10000,
         callbacks=[earlystopper])


# '''    model.fit(x_train, y_train,             ### training
#               batch_size=32,
#               epochs=10000,
#               verbose = 2,
#               validation_split=0.1,
#               shuffle=True,
#               callbacks=[earlystopper])
#    '''
    print(model.evaluate(x_test,y_test))
    print("Trained succsesfully!")
    print("Saving model to JSON")
    json_file = open("block_classifier_model.json", "w") ### exporting model to json
    json_string = model.to_json()
    json_file.write(json_string)
    json_file.close()
    print("Succsess")
    #print("Saving model to PNG")
   # plot_model(model, to_file='model.png')    
    #print("Succsess") 

    print("Saving weights...")                                                  
    model.save_weights("block_classifier_weights.h5")   ### saving weghts
    print("Succsess")   

    return 0

def create_model():
    json_file = open("block_classifier_model.json", "r")
    model_json = json_file.read()    
    model = model_from_json(model_json)
    

    model.load_weights('block_classifier_weights.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model
    
def plot_lastmodel():
    model = create_model()
    print("Saving model to PNG")
    plot_model(model, to_file='model.png',show_shapes=True)    
    print("Succsess")
    return 0
    

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("Input argument!")
        exit()
    if(sys.argv[1] == "-t"):
        training()
        print("END")
        exit()
    if(sys.argv[1] == "-f"):        
        main()
        print("END")
        exit()
    if(sys.argv[1] == "-p"):        
        plot_lastmodel()
        print("END")
        exit()
    


