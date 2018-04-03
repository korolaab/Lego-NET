import PIL
from PIL import Image
import os
import numpy as np


arr_name=[] ###name of classes
def num_class(name): ### quantity of classes
    num = 0
    for i in arr_name:
        
        if(name == i):
            return num
                
        num = num +1

def check_name(name): ### checking class name
    global arr_name
    for i in arr_name:
        if(i == name):
            return False
    return True

'''def name_process(file_name): ### processing name of file if we have 2 '_' we read it until the second '_'
    num = 0                  ### if we have 1 '_' we read until the first '_'
    name = ''
    for i in file_name:
        if(i == '_'):
            num = num + 1
   
    for i in file_name:
        if( i == '_'):
            num = num - 1
        if(num == 0):
            break
        name = name + i
    return name
'''



def load(dataset_folder):               ### main func
    folders = os.listdir(dataset_folder)
    #arr = np.zeros(0)
    #print(files)
    arr =[]
    arr_class = []
    name = ''
    global arr_name
    #num = 0
    for folder in folders:
        files = os.listdir(dataset_folder + '/' + folder)
        if(check_name(folder)):
            arr_name.append(folder)

        print("Woriking with " + dataset_folder+'/'+folder)
        for i in files:
            #print(i)
            im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")  #loading of file and converting to grays
                       
            arr.append(np.asarray(im, dtype="uint8" ))                       #different flips 
            #name = name_process(folder)
            #num = num + 1
            #if(num == 127 or num == 139):
            #    print("Error in:"+ dataset_folder+'/'+folder+'/'+i)
            
            
            arr_class.append(num_class(folder))
                  
        
    arr = np.array(arr)
    arr_class = np.array(arr_class)
    #print(len(arr))
    #print(arr)
    print("=================================================")
    print("I have found {} images seperated into {} classes".format(len(arr),len(arr_name)))
    print("=================================================")
    print(arr_name)    
    return arr,arr_class


#im = (Image.open('dataset' + '/' + 'container' + '/'+'1.gif'))
#image = np.asarray(im.convert("RGB"), dtype="uint8" )
#print(image.shape)
'''
im = (Image.open("1.gif")).convert("RGB")
im.show()  #loading of file and converting to grays
im.transpose(PIL.Image.TRANSPOSE).show()
im.transpose(PIL.Image.FLIP_LEFT_RIGHT).show()
im.transpose(PIL.Image.ROTATE_270).show()
im.transpose(PIL.Image.ROTATE_180).show()
im.transpose(PIL.Image.ROTATE_90).show()
im.transpose(PIL.Image.ROTATE_45).show()
'''


#load("dataset")
