import numpy as np
import sys
import math

import PIL
from PIL import ImageDraw
from PIL import Image
from PIL import ImageFilter

from keras.models import load_model , model_from_json
import tensorflow as tf

import numpy as np
import load_photo as LOAD
import os
import argparse
from matplotlib import pyplot as PLT
from matplotlib import pyplot, transforms
import skimage
from skimage import measure
map = ''
w = ''
Debug = ''


def finder_OpenCV(img):
    for i in range(6):
        if(name[i]=="nothing"):
            continue
        m=img[:,:,i].max()
        img[:,:,i]=img[:,:,i]-m*0.6
    img = np.where(img > 0, 1, 0)
    arr =[]
    # print(img.shape)
    for label in range(6):
        if(name[label]=="nothing"):
            continue
        im = img[:,:,label].T
        contours = measure.find_contours(im, 0.8)
        for contour in contours:
            coordinates = contour.astype(int)
            ymax, xmax = coordinates.max(axis=0)
            ymin, xmin = coordinates.min(axis=0)
            # print("xmin{} ymin{} xmax{} ymax{}".format(xmin,ymin,xmax,ymax))
            cord = (xmin,ymin,xmax-xmin,ymax-ymin)
            objekt = [cord,label]
            arr.append(objekt)
    return arr

def circle_avr(x,y,arr,clas,radius):
    sigma = 0
    n = 0
    for dx in range(x-radius ,x+radius+1):
            for dy in range(round(y -  round(math.sqrt(radius**2-(dx-x)^2))), y+round(math.sqrt(radius**2-(dx-x)^2))):
                if(dx>639 or dy>479):
                    continue
                sigma += arr[dx][dy][clas]
                n+=1
    return sigma/n

def circle_sum(x,y,arr,clas,radius):
    sigma = 0
    n = 0
    for dx in range(x-radius ,x+radius+1):
            for dy in range(round(y -  round(math.sqrt(radius**2-(dx-x)^2))), y+round(math.sqrt(radius**2-(dx-x)^2))):
                if(dx>639 or dy>479):
                    continue
    return sigma

def finder(result): # my finder
    radius = 20
    arr =[]
    global name
    for clas in range(0,6):
        if(name[clas]=="nothing"):
            continue
        max_val=0
        xm = 0
        ym = 0
        objekt=[0,0,0]
        last_val = 0
        for x in range(19):
            for y in range(13):
                dx = x * 30 + 50
                dy = y * 30 + 50
                val = circle_avr(dx,dy,result,clas,radius)
                if val > max_val:
                    max_val = val
                    xm = dx
                    ym = dy
        if max_val >10:
            objekt=[xm,ym,clas]
            arr.append(objekt)
    return arr

def create_model():
    json_file = open("models/cnn_model.json", "r")    ###loading from json file the model
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(w)       ###loading weights from file
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


def check_objects(im,results):
    checked = []
    global name
    global model
    for result in results:
            coord,obj = result### one object
            #x,y,obj = result
            if(name[obj] == "nothing"):
                continue

            x,y,w,h=coord

            sample = np.asarray(im.crop((x, y, x+w, y+h)).convert("RGB").resize((100,100)))
            sample =np.expand_dims(sample, axis=0)/255   ###adding one dimension to sample
            res = model.predict(sample)                  ###neral network prediction

            prob = res[0][obj]

            if(round(prob) != 0 ):
                checked.append(result)


    return checked
def show_im(im,results):
    for result in results:
            coord,obj = result### one object
            #x,y,obj = result
            if(name[obj] == "nothing"):
                continue
            x,y,w,h=coord
            sample = np.asarray(im.crop((x, y, x+w, y+h)).convert("RGB").resize((100,100)))
            sample =np.expand_dims(sample, axis=0)/255   ###adding one dimension to sample
            res = model.predict(sample)                  ###neral network prediction
            prob = res[0][obj]
            print("{} {} {}".format(coord,name[obj],prob))
            draw = ImageDraw.Draw(im)
            draw.rectangle((x,y,x+w,y+h),outline='green')
            draw.text((x,y),text = name[obj]+ "  " + str(prob),fill="green")
            del draw
    im.show()
def process(im):
    global name
    global model
    dx = 0
    dy = 0
    i = 0
    result = []                               ###matrix of results in each piece of image
    heatmap = np.zeros((640,480,6))           ###image
    for x_shift in range(0,541,20):           ###stupid algoritm
        result.append([])
        for y_shift in range(0,381,20):
            dx = x_shift
            dy = y_shift
            sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"))###cropping piece of image
            sample =np.expand_dims(sample, axis=0)/255                               ###adding one dimension to sample
            res = model.predict(sample)                                              ###neral network prediction
            obj = res.argmax()

            if(name[obj] == "nothing"):
                continue


            heatmap[dx:dx+100,dy:dy+100,obj] += res.max()
    return heatmap

def plotting(x):
    fig = PLT.figure()
    for i in range(0,6):
        PLT.subplot(2,3,i+1).set_title(name[i])
        PLT.imshow(x[:,:,i].T,cmap='jet')
        PLT.axis('off')
        PLT.colorbar()
    PLT.show()

def main(im):
    x = process(im)
    y = finder_OpenCV(x)            ### return coordinates of objects[x,y,w,h]

    # y = check_objects(im,y)
    show_im(im,y)
    if(map):
        plotting(x)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finding objects on image")
    parser.add_argument("--image",action = "store", metavar='<path>',default = None, required=True ,dest = "img", help="Image")
    # parser.add_argument("--model",dest='model',default = "cnn", choices=['cnn'], help="Models")
    parser.add_argument("--Weights",action = "store",default="cnn_weights.h5", metavar='<path>',required=True,dest = "w", help="Weights")
    parser.add_argument("-d", "--Debug ",dest='Debug', action="store_false", help="Debuging information")
    parser.add_argument("--map",dest="map",action="store_true",help="Plotting heatmap")
    args = parser.parse_args()
    # print(args.model)
    if(args.Debug):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no debugging info

    name = LOAD.load_names()
    # m = args.model
    w = args.w
    map = args.map
    Debug = args.Debug
    model = create_model()
    main(Image.open(args.img))
