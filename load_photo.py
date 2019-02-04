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

def load_names():
    file = open("labels.txt", "r")
    output = file.read()
    name=[]
    strn=''
    names=[]
    #print(output)
    for i in output:
        if(i != ' '):
            name.append(i)
        if(i == ' '):

            names.append(''.join(name))
            name = []

    return names

#print(num_class(folder))
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
        # if(folder == "nothing"):
        #     continue
        if(check_name(folder)):
            arr_name.append(folder)

       #print("Woriking with " + dataset_folder+'/'+folder)
        for i in files:
            #print(i)
            im = (Image.open(dataset_folder + '/' + folder + '/' +i)).convert("RGB")  #loading of file and converting to RGB

            arr.append(np.asarray(im, dtype="uint8" ))


            #name = name_process(folder)
            #num = num + 1
            #if(num == 127 or num == 139):
            #    print("Error in:"+ dataset_folder+'/'+folder+'/'+i)


            arr_class.append(num_class(folder))


    arr = np.array(arr)
    arr_class = np.array(arr_class)
    print("{} images  ||| {} classes".format(len(arr),len(arr_name)))
    print(arr_name)
    output = open("labels.txt", "w")
    for i in arr_name:
        output.write(i)
        output.write(" ")
    return arr,arr_class
