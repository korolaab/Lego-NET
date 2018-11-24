import numpy as np
import sys
import math
from PIL import ImageDraw
import cv2
from keras.models import load_model , model_from_json
import PIL
from PIL import Image
import numpy as np
    
import load_photo as LOAD
import os
import color as c
#np.set_printoptions(threshold=np.nan) 

name = LOAD.load_names("dataset")

def finder_OpenCV(img):
    for i in range(6):
        if(name[i]=="nothing"):
            continue
        #
        m=img[:,:,i].max()
        print(m)
        img[:,:,i]=img[:,:,i]-m*0.6


    img = np.where(img >0, 1, 0)
    arr =[]
    for label in range(6):

        if(name[label]=="nothing"):
            continue
        im = np.ascontiguousarray(img[:,:,label].T, dtype=np.uint8)
        #print(im.shape)
        # fig = PLT.figure()
        # PLT.imshow(im)
        # PLT.show()
        _, contours, hierarchy = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # contours,hierarchy = cv2.findContours(thresh, 1, 2)
        # im = Image.fromarray(img.astype('uint8'), 'RGB').convert('L')
            #im.show()
            # print(len(contours))
        #print(label)
        #print(contours)
        for cnt in contours:
            cord = cv2.boundingRect(cnt)
            print("{} == {}".format(label,cord))
            objekt = [cord,label]
            arr.append(objekt)
    return arr
def circle_avr(x,y,arr,clas,radius):

    sigma = 0
    n = 0

    for dx in range(x-radius ,x+radius+1):
            for dy in range(round(y -  round(math.sqrt(radius**2-(dx-x)^2))) ,y+round(math.sqrt(radius**2-(dx-x)^2))):
                if(dx>639 or dy>479):
                    continue
                sigma += arr[dx][dy][clas]
                n+=1
    return sigma/n



def circle_sum(x,y,arr,clas,radius):

    sigma = 0
    n = 0

    for dx in range(x-radius ,x+radius+1):
            for dy in range(round(y -  round(math.sqrt(radius**2-(dx-x)^2))) ,y+round(math.sqrt(radius**2-(dx-x)^2))):
                if(dx>639 or dy>479):
                    continue
                sigma += arr[dx][dy][clas]
            
    #print(sigma)
    return sigma
def act(val):
    val = val//10000
    return val/(1+abs(val))

def finder(result):
    #result.sum(0).sum(0)
    # obj_radius = [40,45,45,0,50,50,20,35]

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
                #print(val)
                #if(delta<0 and last_val>5000):



        print("{}={}".format(name[clas],max_val))
        if max_val >10:
            objekt=[xm,ym,clas]
            arr.append(objekt)
    return arr


def create_model():
    json_file = open("block_classifier_model.json", "r")    ###loading from json file the model
    model_json = json_file.read()    
    model = model_from_json(model_json)
    

    model.load_weights('block_classifier_weights.h5')       ###loading weights from file
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model        
model = create_model()

from matplotlib import pyplot as PLT
from matplotlib import pyplot, transforms
import scipy.misc
from scipy import ndimage

def find_best_loc(im,coord,obj):
    global model
    x,y,w,h = coord
    maxprob = 0
    xm=0
    ym=0
    wm=0
    hm=0
    xc = x+round(w/2)
    yc= y+round(h/2)
    for i in range(0,-20,5):
        wt = w/2+i
        ht = h/2+i
        sample = np.asarray(im.crop((xc-wt,y-ht, xc+wt, yc+ht)).convert("RGB").resize((100,100)))
        sample =np.expand_dims(sample, axis=0)/255 ###adding one dimension to sample
        res = model.predict(sample)        ###neral network prediction
        prob = res[0][obj]
        if(prob>maxprob):
            maxprob = prob
            xm=xc-wt
            ym=yc-ht
            wm=2*wt
            hm=2*ht
            
            


    # for i in range(-20,20,5):
    #     for j in range(-50,50,10):
    #         if(x+i<0 or y+i<0 or x+w+j>640 or y+h+j>480):
    #             continue
    #         sample = np.asarray(im.crop((x+i,y+i, x+w+j, y+h+j)).convert("RGB").resize((100,100)))
    #         sample =np.expand_dims(sample, axis=0)/255  ###adding one dimension to sample
    #         res = model.predict(sample)        ###neral network prediction
    #         prob = res[0][obj]
    #         if(prob>maxprob):
    #             maxprob = prob
    #             xm=x+i
    #             ym=y+i
    #             wm=w+j
    #             hm=h+j
    if(xm== 0):
        xm = x
    if(ym== 0):
        ym = y
    if(wm== 0):
        wm = w
    if(hm== 0):
        hm = h


    return xm,ym,wm,hm,maxprob



def image_show(im,results):
    print("")
    print("==========================")
    global name
    global model
    color = c.arr   ### color names
    for result in results:
            coord,obj = result### one object
            #x,y,obj = result
            if(name[obj] == "nothing"):
                continue
            x,y,w,h=coord

            #print(coord)
            
            sample = np.asarray(im.crop((x, y, x+w, y+h)).convert("RGB").resize((100,100)))
            # # PLT.imshow(sample)
            # # PLT.show()
            
            sample =np.expand_dims(sample, axis=0)/255  ###adding one dimension to sample
            res = model.predict(sample)        ###neral network prediction
             
            prob = res[0][obj]
            #print("{} {} {}".format(coord,name[obj],prob))
            #x,y,w,h,prob = find_best_loc(im,coord,obj)

            print("{} {} {}".format(coord,name[obj],prob))
            # if(round(prob)==0):
            #      continue
            draw = ImageDraw.Draw(im)
            draw.rectangle((x,y,x+w,y+h),outline='green')
            draw.text((x,y),text = name[obj]+"  "+str(prob),fill="green")
            del draw
           
    #         k = c.color(im.crop((dx-50, dy-50, dx + 50, dy + 50)).convert("RGB"),obj)
    #         im.paste(k, (dx-50,dy-50,dx + 50,dy+ 50))
        
    # for i in range (0,6):
    #     if(name[i] == "nothing"):
    #             continue

    #     print("{} colored by {}".format(name[i],color[i]))
                                  
    return im
               

from PIL import ImageFilter

def fill(im,x,y,clas,value):
    for dx in range(x ,x+100):
        for dy in range(y ,y+100):
            im[dx][dy][clas]+=value
            #print("{} {} {}".format(im[x][y][clas],x,y))
    return im
def process(im):
    #x_train,y_train = LOAD.load("dataset")
    global name
    global model
    dx = 0
    dy = 0
    i = 0
    result = []                         ###matrix of results in each piece of image
    #ker = 1*np.array([-1,-1,-1,-1,9,-1,-1,-1,-1])
    #print(ker)
    # Convolve
    #im = im.filter(ImageFilter.Kernel((3,3),ker,scale=1,offset=0))
    #im = im.filter(ImageFilter.Kernel((3,3),ker,scale=1,offset=0))
    
    heatmap = np.zeros((640,480,6))           ###image             
    # model = create_model()              ###network model creation
    for x_shift in range(0,541,20):           ###stupid algoritm
        result.append([])
        for y_shift in range(0,381,20):
            dx = x_shift
            dy = y_shift           
            sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"))###cropping piece of image
            sample =np.expand_dims(sample, axis=0)/255  ###adding one dimension to sample
            res = model.predict(sample)        ###neral network prediction
            obj = res.argmax()

            if(name[obj] == "nothing"):
                continue

            
            heatmap[dx:dx+100,dy:dy+100,obj] += res.max()
            #heatmap = fill(heatmap,dx,dy,obj,res.max())

            #if(name[res.argmax()] == "nothing"):
            #    continue
            #print("x = {} y = {}  ::: {} {}".format(dx, dy,name[res.argmax()],res.max()))
        
    #result = np.array(result)
    return heatmap

def plotting(x):
    #x = x.reshape(480,640,8)
    fig = PLT.figure()
    for i in range(0,6):
        PLT.subplot(2,3,i+1).set_title(name[i])

        
        PLT.imshow(x[:,:,i].T,cmap='jet')
        
        PLT.axis('off')
        PLT.colorbar()

    # PLT.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    # cax = PLT.axes([0.85, 0.1, 0.075, 0.8])
    # PLT.colorbar(cax=cax)
    
    
    PLT.show()



def main(arg):   
       ###names of objects from folders in "dataset/"
    
    im = Image.open(arg)             ###load image which was set in arguments
    
    x = process(im)
    
    y = finder_OpenCV(x)
    
    # PLT.imshow(x[:,:,2])
    # PLT.show()
    

    im = image_show(im,y)

    im.show()
    
    plotting(x)
    return y
if __name__ == '__main__':
    if(len(sys.argv) < 2):          ###parsing arguments
        print("Input path to the image!")
        exit()

    os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"
    name = LOAD.load_names("dataset")
    #model = create_model()
    #pre = model.prediction(sys.argv[1])
    #print(pre.argmax())
    #print(pre)
    #LOAD.load("dataset")
    #print(name)
    main(sys.argv[1])
