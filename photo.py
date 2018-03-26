import os
import cv2
import main


def convert(folder,save):
    files = os.listdir(folder)
    print(files)
    x = 0
    for file in files:
        print("working with{}".format(file))
        main.image_mode(cv2.imread(folder+file),save,x)
        x = x + 1
