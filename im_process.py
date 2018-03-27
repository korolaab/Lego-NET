import PIL
from PIL import Image
import numpy as np

def process(im):
	dx = 0
	dy = 0
	i = 0
	result = []
	for x_shift in range(55):
		result.append([])
		for y_shift in range(39):
			dx = x_shift * 10
			dy = y_shift * 10			
			sample = np.asarray(im.crop((dx, dy, dx + 100, dy + 100)).convert("L"))
			result[x_shift].append("{} = {}".format(dx,dy)) ### temporary
			#print("photo")			
			#print("dx = {}   dy = {}".format(dx,dy))
	#print(result[51][11])erq




#im = Image.open("1.jpg")

#process(im)