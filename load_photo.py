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

def name_process(file_name): ### processing name of file if we have 2 '_' we read it until the second '_'
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
   
def load(folder):               ### main func
    files = os.listdir(folder)
    #arr = np.zeros(0)
    #print(files)
    arr =[]
    arr_class = []
    name = ''
    global arr_name
    for i in files:        
        im = (Image.open(folder + '/' + i)).resize([100,100],resample=0).convert("L")   #loading of file
                   
        arr.append(np.asarray(im, dtype="uint8" ))                          #different flips 
        arr.append(np.asarray(im.transpose(PIL.Image.TRANSPOSE), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.FLIP_LEFT_RIGHT), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.FLIP_TOP_BOTTOM), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.ROTATE_180), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.ROTATE_270), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.ROTATE_90), dtype="uint8" ))
        name = name_process(i)
        
        if(check_name(name)):
            arr_name.append(name)
        num = 7
        while num > 0:
            arr_class.append(num_class(name))
            num = num - 1        
        
    arr = np.array(arr)
    arr_class = np.array(arr_class)
    #print(len(arr))
    #print(arr)
    print("=================================================")
    print("I have found {} images seperated into {} classes".format(len(arr),len(arr_name)))
    print("=================================================")
    print(arr_name)    
    return arr,arr_class

