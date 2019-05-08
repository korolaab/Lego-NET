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
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def finder_OpenCV(img): # OpenCV contours finder
    for i in range(6):
        if(name[i]=="nothing"):
            continue
        m=img[:,:,i].max()
        # img[:,:,i]=img[:,:,i]-m*0.6
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

def circle_avr(x,y,arr,clas,radius): # calculates the average of the elements inside the circle in the matrix
    sigma = 0
    n = 0
    for dx in range(x-radius ,x+radius+1):
            for dy in range(round(y -  round(math.sqrt(radius**2-(dx-x)^2))), y+round(math.sqrt(radius**2-(dx-x)^2))):
                if(dx>639 or dy>479):
                    continue
                sigma += arr[dx][dy][clas]
                n+=1
    return sigma/n


def circle_sum(x,y,arr,clas,radius):  # calculates the sum of the elements inside the circle in the matrix
    sigma = 0
    n = 0
    for dx in range(x-radius ,x+radius+1):
            for dy in range(round(y -  round(math.sqrt(radius**2-(dx-x)^2))), y+round(math.sqrt(radius**2-(dx-x)^2))):
                if(dx>639 or dy>479):
                    continue
    return sigma

def finder(result): # self created finder
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

def load_model(m,weights=None):
    json_file = open("models/{}_model.json".format(m), "r")    ###loading from json file the model
    model_json = json_file.read()
    model = model_from_json(model_json)
    if(D):
        model.summary()
    if(weights is None):
        return model
    model.load_weights(weights)       ###loading weights from file
    return model

def show_im(im,results):
    img = im.convert("RGB")
    for result in results:
            coord,obj = result### one object
            #x,y,obj = result
            if(name[obj] == "nothing"):
                continue
            x,y,w,h=coord
            sample = np.asarray(img.crop((x, y, x+w, y+h)).convert("RGB").resize((100,100)))
            sample =np.expand_dims(sample, axis=0)/255   ###adding one dimension to sample
            res = model.predict(sample)                  ###neral network prediction
            prob = res[0][obj]
            print("{} {} {}".format(coord,name[obj],prob))
            if(round(prob) == 0 ):
                continue
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





def cnn_model(im):
    x = process(im)
    # y = finder_OpenCV(x)            ### return coordinates of objects[x,y,w,h]
    # y = check_objects(im,y)
    # show_im(im,y)

    fig = PLT.figure()
    for i in range(0,6):
        PLT.subplot(2,3,i+1).set_title(name[i])
        PLT.imshow(x[:,:,i].T)
        PLT.axis('off')
        # PLT.colorbar()
    PLT.subplot(2,3,6).set_title("photo")
    PLT.imshow(im)
    return 0
def unet_model(im):
    name = ['block_6x1', 'container', 'engine', 'wheel_big', 'wheel_middle']
    img = np.asarray(im.convert("RGB"))
    im = np.expand_dims(img,axis=0)/255 #U-net input is (1,480,640,3)
    x = model.predict(im)
    fig = PLT.figure(1)
    for i in range(0,5): #  show plot for all prediction
        PLT.subplot(2,3,i+1).set_title(name[i])
        PLT.axis('off')
        PLT.imshow(x[0,:,:,i])
    PLT.subplot(2,3,6).set_title("photo")
    PLT.axis('off')
    PLT.imshow(img)
    return 0
name = LOAD.load_names()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finding objects on image")
    parser.add_argument("--image",action = "store", metavar='<path>',default = None, required=True ,dest = "img", help="Image")
    parser.add_argument("--model",dest='model',default = "cnn", choices=["cnn","unet"], help="Models")
    parser.add_argument("--weights",action = "store", metavar='<path>', dest = "w", help="Weights")
    parser.add_argument("-d", "--Debug ",dest='Debug', action="store_true", help="Debuging information and models parameters")
    args = parser.parse_args()
    m = args.model
    D= args.Debug
    w = args.w
    if(not D):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###no TF debugging info


    if(w == None):
        print(bcolors.WARNING + "No pretrained weights the results can be unpredictable"+ bcolors.ENDC)

    if(m == "cnn"):
        model = load_model("cnn",weights = w)
        cnn_model(Image.open(args.img))

    if(m == "unet"):
        model = load_model("unet",weights = w)
        unet_model(Image.open(args.img))
    # PLT.show()
