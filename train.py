import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.utils import np_utils 
from keras.models import load_model, model_from_json
import os
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
import load_photo as LOAD
import sys
from keras.utils import plot_model




def data():
    num_classes = len(LOAD.arr_name)
    x_train,y_train = LOAD.load("dataset")   #load photos and answers 
    x_train = x_train.astype('float32')
    x_train/= 255
    y_train = np_utils.to_categorical(y_train,num_classes)
    return x_train,y_train





#x_train = np.expand_dims(x_train, axis=3)
#print(x_train.shape)





###============================================================### model
def train_model(x_train,y_train):
    batch_size = {{choice([32, 64])}}
    epochs = 30
    num_classes = len(LOAD.arr_name)
    Input_main = Input(shape=(100,100,3), dtype='float32', name='Input_main')
    convolution_matrix = {{choice([3, 5])}}
    pooling_matrix = {{choice([2, 4])}} 
    Conv1 = Conv2D({{choice([32, 64])}}, (convolution_matrix, convolution_matrix),activation = {{choice(['relu', 'sigmoid'])}},name="conv1")(Input_main)
    Pool1 = MaxPooling2D(pool_size=(pooling_matrix, pooling_matrix),name = "pool1")(Conv1)
    Drop1 = Dropout({{uniform(0, 1)}},name="drop1")(Pool1)
    
    con = conditional({{choice(['one','two'])}})

    if con == "two":
        Conv2 = Conv2D({{choice([32, 64])}}, (convolution_matrix, convolution_matrix),activation = {{choice(['relu', 'sigmoid'])}},name="conv2")(Drop1)
        Pool2 = MaxPooling2D(pool_size=(pooling_matrix, pooling_matrix),name = "pool2")(Conv2)
        Drop2 = Dropout({{uniform(0, 1)}},name="drop2")(Pool2)
        Flat = Flatten(name = "Flat")(Drop2) 
    else:
        Flat = Flatten(name = "Flat")(Drop1)    
    '''    
    elif con == "three":
        Conv2 = Conv2D({{choice([32, 64, 128])}}, (convolution_matrix, convolution_matrix),activation = "relu",name="conv2")(Drop1)
        Pool2 = MaxPooling2D(pool_size=(pooling_matrix, pooling_matrix),name = "pool2")(Conv2)
        Drop2 = Dropout({{uniform(0, 1)}},name="drop2")(Pool2 )
        
        Conv3 = Conv2D({{choice([32, 64, 128])}}, (convolution_matrix, convolution_matrix),activation = "relu",name="conv3")(Drop2)
        Pool3 = MaxPooling2D(pool_size=(pooling_matrix, pooling_matrix),name = "pool3")(Conv3)
        Drop3 = Dropout({{uniform(0, 1)}},name="drop3")(Pool3)
        Flat = Flatten(name = "Flat")(Drop3)'''
             
               

    

    

    Dens1 = Dense({{choice([16 ,32, 64, 256])}},activation = {{choice(['relu', 'sigmoid'])}},name = "dens1")(Flat)
    Drop_last = Dropout({{uniform(0, 1)}},name = "drop_last")(Dens1)

  

    OUT = Dense(num_classes,activation = "softmax",name = "output")(Drop_last)

    model = Model(inputs=Input_main, outputs=OUT)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    

    model.summary()     
    
    model.fit(x_train, y_train,             ### training
              batch_size=batch_size,
              epochs=epochs,
              verbose = 2,
              validation_split=0.1,
              shuffle=True)
    score = model.evaluate(x_train, y_train, verbose=1)
    print(score)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
###===================================================================### end model
def main():
    best_run,best_model = optim.minimize(model=train_model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=20,
                              trials=Trials())
    x_train,y_train = data()

    print("Trained succsesfully!")
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_train,y_train))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Saving model to JSON")
    json_file = open("block_classifier_model.json", "w") ### exporting model to json
    json_string = best_model.to_json()
    json_file.write(json_string)
    json_file.close()
    print("Succsess") 

    print("Saving weights...")                                                  
    best_model.save_weights("block_classifier_weights.h5")   ### saving weghts
    print("Succsess")   
    return 0


def training():
    batch_size = 32
    epochs = 200
    json_file = open("block_classifier_model.json", "r")
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('block_classifier_weights.h5')
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    x_train, y_train = data()
    model.fit(x_train, y_train,             ### training
              batch_size=batch_size,
              epochs=epochs,
              verbose = 2,
              validation_split=0.1,
              shuffle=True)

    print("Trained succsesfully!")
    print("Saving model to JSON")
    json_file = open("block_classifier_model.json", "w") ### exporting model to json
    json_string = model.to_json()
    json_file.write(json_string)
    json_file.close()
    print("Succsess")
    print("Saving model to PNG")
    plot_model(model, to_file='model.png')    
    print("Succsess") 

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
    


