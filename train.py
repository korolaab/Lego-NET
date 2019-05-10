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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import PIL
from PIL import ImageDraw
from PIL import Image
from PIL import ImageFilter
import scipy.misc
from scipy.ndimage import rotate
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import random
def val_load(dataset_folder):
    photos = []
    # print(type(photos))
    answ = []
    folders = os.listdir(dataset_folder)
    x = 0
    for folder in folders:
            mask=[]
            # folder = is i
            files = os.listdir(dataset_folder + '/' + folder)
            n=1
            files = sorted(files)
            # print(files)
            for i in files:
                if( i[:-4]=="photo"):
                    im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")
                    photo = np.array(im,dtype = "float32")

                    photo = photo/255
                    # photo = np.expand_dims(photo,axis=0)
                else:
                    im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("L")
                    mask.append(np.asarray(im,dtype="float32")/255)
                    n+1
            mask = np.array(mask)
            mask = np.swapaxes(mask,0,2)
            mask = np.swapaxes(mask,0,1)
            # mask = np.expand_dims(mask,axis=0)
            mask[mask > 0] = 1
            # print(type(photos))
            photos.append(photo)
            answ.append(mask)
            # x = x+1
            # if(x == 2):
            #     break
    photos = np.array(photos)
    answ = np.array(answ)
    return photos,answ

def gen(dataset_folder,batch_size):
    folders = os.listdir(dataset_folder)
    n = 0
    while True:
        mask=[]

        folder = random.choice(folders)
        # folder = folders[n]
        files = os.listdir(dataset_folder + '/' + folder)
        # n=1
        v_f = bool(random.getrandbits(1))
        g_f = bool(random.getrandbits(1))
        bri = random.uniform(0.5, 1)
        con = random.uniform(0.5, 1)
        col = random.uniform(0, 1)

        blur = random.uniform(0, 1)
        # noise = random.randint(1, 60)

        r = random.uniform(-5,5)
        files = sorted(files)
        # print(files)
        for i in files:
            if( i[:-4]=="photo"):
                im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")
                im = PIL.ImageEnhance.Brightness(im).enhance(bri)
                im = PIL.ImageEnhance.Contrast(im).enhance(con)
                im = PIL.ImageEnhance.Color(im).enhance(col)
                im = PIL.ImageEnhance.Sharpness(im).enhance(blur)

                photo = np.asarray(im,dtype="int32")
                # photo = np.array(im,dtype = "int32")
                if(v_f):
                     photo = photo[::-1, :]
                if(g_f):
                     photo = photo[:, ::-1]
                # photo = ndimage.uniform_filter(photo, size=(blur, blur, 1))
                # # photo = np.array(photo,dtype = "float32")
                # photo = np.random.normal(2*photo+2,noise)
                photo = rotate(photo,r,reshape=False)
                # print(photo)
                photo = np.array(photo,dtype = "float32")/255
                photo = np.expand_dims(photo,axis=0)
            else:
                im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("L")
                m = np.asarray(im,dtype="float32")
                if(v_f):
                     m = m[::-1, :]
                if(g_f):
                     m = m[:, ::-1]
                m = rotate(m,r,reshape=False)
                # m = m255
                mask.append(m/255)
        mask = np.array(mask)
        mask = np.swapaxes(mask,0,2)
        mask = np.swapaxes(mask,0,1)
        mask = np.expand_dims(mask,axis=0)
        mask = np.around(mask)

        yield photo,mask


def d_c(y_true, y_pred):
    y_true.set_shape(y_pred.get_shape())

    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)

def dice_coef_loss(y_true, y_pred):
        return 1 - d_c(y_true,y_pred)

def data_cnn(dataset,test_dataset):
    # print(dataset)
    num_classes = len(LOAD.arr_name)
    x_train,y_train = LOAD.load(dataset)   #load photos and answers
    x_train = x_train.astype('float32')
    y_train = np_utils.to_categorical(y_train,num_classes)
    x_test,y_test = LOAD.load(test_dataset)
    x_test = x_test.astype('float32')/255
    y_test = np_utils.to_categorical(y_test,num_classes)

    datagen=ImageDataGenerator(
                        rescale=1./255,
                        brightness_range=(0.5,1),
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        channel_shift_range=0.01,
                        rotation_range=0.01,
                        zoom_range=0.01)

    return datagen, x_train,y_train,x_test,y_test

def load_model(m,weights=None):
    json_file = open("models/{}_model.json".format(m), "r")    ###loading from json file the model
    model_json = json_file.read()
    model = model_from_json(model_json)
    if(weights is None):
        return model
    model.load_weights(weights)       ###loading weights from file
    return model

import matplotlib.pyplot as plt
def cnn_training(model,w = None):
    model = load_model(model,w)
    model.summary()
    d,x_train,y_train,x_test,y_test = data_cnn("dataset","test_dataset")
    model.compile(optimizer="adam",loss="categorical_crossentropy",
                    metrics=["accuracy"])
    batch_size = 32
    сheckpoint = ModelCheckpoint('save/cnn_weights-{epoch:02d}-{loss:.4f}.hdf5',period = 50)
    history = model.fit_generator(d.flow(x_train,y_train),
                        steps_per_epoch = len(x_train)//batch_size,
                        validation_data = (x_test,y_test),
                        epochs = 300,
                        callbacks=[сheckpoint])
    model.save_weights("models/cnn_weights.h5")
    return 0

def unet_training(model,w = None):
    model = load_model(model,w)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[d_c])
    сheckpoint = ModelCheckpoint('save/unet_weights-{epoch:02d}-{loss:.4f}.hdf5',period = 50)
    x_val,y_val = val_load("unet_validate")
    model.fit_generator(gen("dataset_fcn",1),
                    steps_per_epoch = 25,
                    validation_data=(x_val,y_val),
                    epochs = 10000,
                    callbacks=[сheckpoint])
    model.save_weights("models/unet_weights.h5")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training a model")
    parser.add_argument("--model",dest='model', choices=['cnn','unet'], required=True, help="Models")
    parser.add_argument("-w","--weights",action = "store", metavar='<path>',default = None,dest = "w", help="Pretrained weights")
    parser.add_argument("-d", "--Debug ",dest='Debug', action="store_true", help="Tensorflow debuging information")
    args = parser.parse_args()
    m = args.model
    w = args.w
    if(args.Debug):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no debugging info TF
    if(m == "cnn"):
        cnn_training(m,w)

    if(m == "unet"):
        unet_training(m,w)
