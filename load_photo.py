import PIL
from PIL import Image
import os
import numpy as np


arr_name=[]
def num_class(name):
    num = 0
    for i in arr_name:
        
        if(name == i):
            return num
                
        num = num +1 
def check_name(name):
    global arr_name
    for i in arr_name:
        if(i == name):
            return False
    return True

def name_process(file_name):
    num = 0
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

transpose = ['','','','']
    
def load(folder):
    files = os.listdir(folder)
    #arr = np.zeros(0)
    #print(files)
    arr =[]
    arr_class = []
    name = ''
    global arr_name
    for i in files:
        num = 7
        im = (Image.open(folder + '/' + i)).resize([100,100],resample=0).convert("L")
                   
        arr.append(np.asarray(im, dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.TRANSPOSE), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.FLIP_LEFT_RIGHT), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.FLIP_TOP_BOTTOM), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.ROTATE_180), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.ROTATE_270), dtype="uint8" ))
        arr.append(np.asarray(im.transpose(PIL.Image.ROTATE_90), dtype="uint8" ))
        name = name_process(i)
        
        if(check_name(name)):
            arr_name.append(name)
        while num > 0:
            arr_class.append(num_class(name))
            num = num - 1        
        
    arr = np.array(arr)
    arr_class = np.array(arr_class)
    #print(len(arr))
    #print(arr)
    print("================================================")
    print("I have found {} images seperated into {} classes".format(len(arr),len(arr_name)))
    print("================================================")
    print(arr_name)    
    return arr,arr_class 










#print(name_process("plate_6x10_1.jpg"))
#load("/home/alex/Documents/Robo_hand/photos")
'''
im = Image.open("/home/alex/Documents/Robo_hand/photos/container_1.jpg")
im.show()
im.transpose(PIL.Image.TRANSPOSE).show()
im.transpose(PIL.Image.FLIP_LEFT_RIGHT).show()
im.transpose(PIL.Image.FLIP_TOP_BOTTOM).show()
im.transpose(PIL.Image.ROTATE_180).show()
im.transpose(PIL.Image.ROTATE_270).show()
im.transpose(PIL.Image.ROTATE_90).show()

im2 = Image.open("/home/alex/Documents/Robo_hand/photos/container_2.jpg")
im = (im.convert("L")).resize([100,100],resample=0)
im2 = (im2.convert("L")).resize([100,100],resample=0)


pix = np.array(im)
pix2 = np.array(im2)
pix = np.stack((pix,pix2))
print(pix)
im.show()
im2.show()'''
