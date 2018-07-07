
import sys

    


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
    for x_shift in range(55):
        for y_shift in range(39):
            num = results[x_shift][y_shift].argmax() ### coordinate of square (x_shift,y_shift)
            if(name[num] == "nothing"):
                continue
            if(num > 3):
                num-=1
            dx = x_shift * 10
            dy = y_shift * 10             
            k = c.color(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"),num)
            im.paste(k, (dx,dy,dx + 100,dy+ 100))
    i = 0    
    while i < 7 :
        if(name[i] == "nothing"):
                print("nothing has no color")
                
        if(i > 3):
            print("{} colored by {}".format(name[i],color[i-1]))
        else:
            print("{} colored by {}".format(name[i],color[i]))
        i = i + 1                          
    return im
               

              
    
def process(im):
    #x_train,y_train = LOAD.load("dataset")
    global name
    dx = 0
    dy = 0
    i = 0
    result = []                         ###matrix of results in each piece of image
    model = create_model()              ###network model creation
    for x_shift in range(55):           ###stupid algoritm
        result.append([])
        for y_shift in range(39):
            dx = x_shift * 10
            dy = y_shift * 10			
            sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("RGB"))###cropping piece of image
            sample =np.expand_dims(sample, axis=0)  ###adding one dimension to sample
            res = model.predict(x = sample)         ###neral network prediction
            result[x_shift].append(res)
            if(name[res.argmax()] == "nothing"):
                continue
            print("x = {} y = {}  ::: {}".format(dx, dy,name[res.argmax()]))
            #print(res)
            #print(name)
            
			#print("photo")			
			#print("dx = {}   dy = {}".format(dx,dy))
	#print(result[51][11])erq
    return result    

if __name__ == '__main__':
    
    if(len(sys.argv) < 2):          ###parsing arguments
        print("Input path to the image!")
        exit()
    import PIL
    from PIL import Image
    import numpy as np
    from keras.models import load_model , model_from_json
    import load_photo as LOAD
    
    import color as c
    name = LOAD.load_names("dataset")          ###names of objects from folders in "dataset/"
    im = Image.open(sys.argv[1])               ###load image which was set in arguments
    
    x = process(im)
    im = image_show(im,x)
    print (name)    
    im.show()
    #print(x)
