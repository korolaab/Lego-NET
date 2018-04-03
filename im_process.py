import PIL
from PIL import Image
import numpy as np
from keras.models import load_model , model_from_json
import load_photo as LOAD
import sys


  
def create_model():
    json_file = open("block_classifier_model.json", "r")
    model_json = json_file.read()    
    model = model_from_json(model_json)
    

    model.load_weights('block_classifier_weights.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model        
        
    
def process(im):
    x_train,y_train = LOAD.load("dataset")
    name = LOAD.arr_name 
    dx = 0
    dy = 0
    i = 0
    result = []
    model = create_model()
    for x_shift in range(55):
        result.append([])
        for y_shift in range(39):
            dx = x_shift * 10
            dy = y_shift * 10			
            sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"))
            sample =np.expand_dims(sample, axis=0)
			##result[x_shift].append("{} = {}".format(dx,dy)) ### temporary
            res = model.predict(x = sample)
            result[x_shift].append(res)
            if(name[res.argmax()] == "nothing"):
                continue
            print("x = {} y = {}  ::: {}".format(dx, dy,name[res.argmax()] ))
            #print(res)
            #print(name)
            
			#print("photo")			
			#print("dx = {}   dy = {}".format(dx,dy))
	#print(result[51][11])erq
    return result    

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("Input path to the image!")
        exit()
    im = Image.open(sys.argv[1])

    x = process(im)
    #print(x)
