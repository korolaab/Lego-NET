import numpy as np
import sys


np.set_printoptions(threshold=np.nan)   

<<<<<<< HEAD
def circle_sum(x,y,arr,clas,radius):	
	sigma = 0
	for dx in range(x ,x+100):
			for dy in range(y ,y+100):
=======
def circle_sum(x,y,arr,clas,radius):

	sigma = 0
	for dx in range(x-radius ,x+radius+1):
			for dy in range(y-radius ,y+radius+1):
				if(dx>639 or dy>479):
					continue
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
				sigma += arr[dx][dy][clas]
	#print(sigma)
	return sigma
def finder(result):
	#result.sum(0).sum(0)
<<<<<<< HEAD
	arr =[]
	for clas in range(0,8):		
=======
	obj_radius = [40,45,45,0,50,50,20,35]
	arr =[]
	global name
	for clas in range(0,8):
		if(name[clas]=="nothing"):
			continue		
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
		max_delta=0
		xm = 0
		ym = 0
		objekt=[0,0,0]
<<<<<<< HEAD
		for x in range(55):
			for y in range(38):
				dx = x *10
				dy = y * 10
				delta = circle_sum(dx,dy,result,clas,radius=50)
				if delta > max_delta:
					max_delta = delta
					
					xm = dx
					ym = dy
					#print(max_delta)
					#print(dx)
					#print(dy)
			#print('1')
		objekt=[xm+50,ym+50,clas]
		#print('22222222222222222222222222')
		arr.append(objekt)
=======
		for x in range(19):
			for y in range(13):
				dx = x * 30 + 50
				dy = y * 30 + 50
				delta = circle_sum(dx,dy,result,clas,obj_radius[clas])
				if delta > max_delta:
					max_delta = delta					
					xm = dx
					ym = dy
		print("{}={}".format(name[clas],round(max_delta)))
		if max_delta >1000:
			objekt=[xm,ym,clas]
			arr.append(objekt)
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
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
            dx,dy,obj = result### one object
            if(name[obj] == "nothing"):
                continue
<<<<<<< HEAD
            if(obj > 3):
                obj-=1             
            k = c.color(im.crop((dx-50, dy-50, dx + 50, dy + 50)).convert("RGB"),obj)
            im.paste(k, (dx-50,dy-50,dx + 50,dy+ 50))
    i = 0    
    while i < 7 :
=======
             
            k = c.color(im.crop((dx-50, dy-50, dx + 50, dy + 50)).convert("RGB"),obj)
            im.paste(k, (dx-50,dy-50,dx + 50,dy+ 50))
        
    for i in range (0,8):
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
        if(name[i] == "nothing"):
                continue

        print("{} colored by {}".format(name[i],color[i]))
                                  
    return im
               

              
<<<<<<< HEAD
def fill(im,x,y,clas):
	for dx in range(x ,x+100):
		for dy in range(y ,y+100):
			im[dx][dy][clas]+=1
=======
def fill(im,x,y,clas,value):
	for dx in range(x ,x+100):
		for dy in range(y ,y+100):
			im[dx][dy][clas]+=value
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
			#print("{} {} {}".format(im[x][y][clas],x,y))
	return im
def process(im):
    #x_train,y_train = LOAD.load("dataset")
    global name
    dx = 0
    dy = 0
    i = 0
    result = []							###matrix of results in each piece of image
    image = np.zeros((640,480,8))           ###image             
    model = create_model()              ###network model creation
    for x_shift in range(55):           ###stupid algoritm
        result.append([])
        for y_shift in range(39):
            dx = x_shift * 10
            dy = y_shift * 10			
            sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"))###cropping piece of image
            sample =np.expand_dims(sample, axis=0)  ###adding one dimension to sample
<<<<<<< HEAD
            res = model.predict(x = sample)        ###neral network prediction
            if(name[res.argmax()] == "nothing"):
                continue
            image = fill(image,dx,dy,res.argmax())

            #if(name[res.argmax()] == "nothing"):
            #    continue
            #print("x = {} y = {}  ::: {}{}".format(dx, dy,name[res.argmax()],res.argmax()))
=======
            res = model.predict(sample)        ###neral network prediction
            obj = res.argmax()
            if(name[obj] == "nothing"):
                continue

            #print(res[obj])
            image = fill(image,dx,dy,obj,res.max())

            #if(name[res.argmax()] == "nothing"):
            #    continue
            print("x = {} y = {}  ::: {}{}".format(dx, dy,name[res.argmax()],res.argmax()))
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83

    #result = np.array(result)
    return image

def main(arg):   
       ###names of objects from folders in "dataset/"
<<<<<<< HEAD
    
    im = Image.open(arg)             ###load image which was set in arguments
    
=======
    
    im = Image.open(arg)             ###load image which was set in arguments
    
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
    x = process(im)
    print(x.shape)
    y = finder(x)
    print(y)
    im = image_show(im,y)

    im.show()
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
<<<<<<< HEAD
    main(sys.argv[1])
=======
    model = create_model()
    pre = model.prediction(sys.argv[1])
    print(pre.argmax())
    print(pre)
    #LOAD.load("dataset")
    #print(name)
   	#main(sys.argv[1])
>>>>>>> 53f561826127f1d84b094b3b3d9c96acc748fd83
