import numpy as np
import sys
import math
from PIL import ImageDraw
import cv2

np.set_printoptions(threshold=np.nan) 



def finder_OpenCV(img):
    img = np.where(img > 40, 255, 0)
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

def circle_sum(x,y,arr,clas,radius):

	sigma = 0
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

				val = circle_sum(dx,dy,result,clas,radius)
				if val > max_val:
					max_val = val					
					xm = dx
					ym = dy
				delta = val - last_val
				#print(delta)
				last_val = val
                #if(delta<0 and last_val>5000):



		print("{}={} = activation {}".format(name[clas],round(max_val),act(max_val)))
		if max_val >1000:
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

def image_show(im,results):
    print("")
    print("==========================")
    global name
    color = c.arr   ### color names
    for result in results:
            #coord,obj = result### one object
            x,y,obj = result
            if(name[obj] == "nothing"):
                continue
            #x,y,w,h=coord

            #print(coord)

            draw = ImageDraw.Draw(im)
            draw.point((x,y),fill='green')
            draw.text((x,y),text = name[obj],fill="green")
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
    dx = 0
    dy = 0
    i = 0
    result = []							###matrix of results in each piece of image
    #ker = 1*np.array([-1,-1,-1,-1,9,-1,-1,-1,-1])
    #print(ker)
# Convolve
    #im = im.filter(ImageFilter.Kernel((3,3),ker,scale=1,offset=0))
    #im = im.filter(ImageFilter.Kernel((3,3),ker,scale=1,offset=0))
    
    image = np.zeros((640,480,6))           ###image             
    model = create_model()              ###network model creation
    for x_shift in range(55):           ###stupid algoritm
        result.append([])
        for y_shift in range(39):
            dx = x_shift * 10
            dy = y_shift * 10			
            sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"))###cropping piece of image
            sample =np.expand_dims(sample, axis=0)/255  ###adding one dimension to sample
            res = model.predict(sample)        ###neral network prediction
            obj = res.argmax()

            if(name[obj] == "nothing"):
                continue

            
           
            image = fill(image,dx,dy,obj,res.max())

            #if(name[res.argmax()] == "nothing"):
            #    continue
            print("x = {} y = {}  ::: {} {}".format(dx, dy,name[res.argmax()],res.max()))
        
    #result = np.array(result)
    return image
from matplotlib import pyplot as PLT
from matplotlib import pyplot, transforms
import scipy.misc
from scipy import ndimage
def plotting(x):
    #x = x.reshape(480,640,8)
    fig = PLT.figure()
    for i in range(0,6):
        PLT.subplot(2,3,i+1).set_title(name[i])

        
        PLT.imshow(x[:,:,i].T,cmap='jet',vmin=0, vmax=100)
        
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
    
    y = finder(x)
    
    # PLT.imshow(x[:,:,2])
    # PLT.show()
    

    im = image_show(im,y)

    im.show()
    
    plotting(x)
if __name__ == '__main__':
    if(len(sys.argv) < 2):          ###parsing arguments
        print("Input path to the image!")
        exit()
    import PIL
    from PIL import Image
    import numpy as np
    from keras.models import load_model , model_from_json
    import load_photo as LOAD
    import os
    import color as c
    os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"
    name = LOAD.load_names("dataset")
    #model = create_model()
    #pre = model.prediction(sys.argv[1])
    #print(pre.argmax())
    #print(pre)
    #LOAD.load("dataset")
    #print(name)
    main(sys.argv[1])
