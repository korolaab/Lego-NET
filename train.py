import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.models import load_model, model_from_json#
import os
import load_photo as LOAD
import sys
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from random import *
import tensorflow as tf
from keras.utils import np_utils
import argparse
def data(dataset,test_dataset):
    print(dataset)
    num_classes = len(LOAD.arr_name)
    x_train,y_train = LOAD.load(dataset)   #load photos and answers
    x_train = x_train.astype('float32')
    y_train = np_utils.to_categorical(y_train,num_classes)
    x_test,y_test = LOAD.load(test_dataset)
    x_test = x_test.astype('float32')/255
    y_test = np_utils.to_categorical(y_test,num_classes)

    datagen=ImageDataGenerator(
                        rescale=1./255,
                        brightness_range=(0.4,1),
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        channel_shift_range=0.1,
                        rotation_range=0.1,
                        zoom_range=0.1)

    return datagen, x_train,y_train,x_test,y_test
def cnn():
    json_file = open("models/cnn_model.json", "r")    ###loading from json file the model
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_model(model, pretrained_weights = None):
    if(model == "cnn"):
        model = cnn()
    if(pretrained_weights):
    	model.load_weights(pretrained_weights) ###loading weights from file
    return model

def training(model,w = None):
    model = create_model(model,w)
    d,x_train,y_train,x_test,y_test = data("dataset","test_dataset")
    batch_size = 32
    model.fit_generator(d.flow(x_train,y_train),
                        steps_per_epoch = len(x_train)//batch_size,
                        validation_data = (x_test,y_test),
                        epochs = 100)
    model.save_weights("model_weights.h5")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training a model")
    parser.add_argument("--model",dest='model', choices=['cnn'], required=True, help="Models")
    parser.add_argument("-w","--Weights",action = "store", metavar='<path>',default = None,dest = "w", help="Pretrained weights")
    parser.add_argument("-d", "--Debug ",dest='Debug', action="store_true", help="Tensorflow's debuging information")
    args = parser.parse_args()
    print(args.model)
    if(args.Debug):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no debugging info TF
    training(args.model,args.w)
