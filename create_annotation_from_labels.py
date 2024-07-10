
import os
import cv2
from PIL import Image
import numpy as np
def xywh_x1y1x2y2(val,iw,ih):
	c = int(val[0])
	x = val[1]*iw
	y = val[2]*ih
	w = val[3]*iw
	h = val[4]*ih
	x1 = int(x-w/2)
	x2 = int(x+w/2)
	y1 = int(y-h/2)
	y2 = int(y+h/2)
	return [x1,y1,x2,y2,c]

def from_labels_to_annotation(list_file,labels_dir,images_dir):
	names = os.listdir(labels_dir)
	for n in names:
		label_path = os.path.join(labels_dir,n)
		image_path = os.path.join(images_dir,n.replace("txt","jpg"))
		list_file.write(image_path)
		print(label_path,image_path)
		val = np.loadtxt(label_path)
		image = Image.open(image_path)
		iw, ih = image.size
		
		
		if(len(val.shape) == 1):
			if(val.shape[0] == 0):
				continue
			coord_c = xywh_x1y1x2y2(val,iw,ih)
			list_file.write(" " + ",".join([str(a) for a in coord_c]) )
			list_file.write('\n')
			continue
			
			
		for i in range(len(val)):
			v = val[i]
			coord_c = xywh_x1y1x2y2(v,iw,ih)
			list_file.write(" " + ",".join([str(a) for a in coord_c]) )
		list_file.write('\n')	
		
if __name__ == "__main__":
	list_file = open('./annotation2.txt', 'w')
	labels_dir = "./labels/" 
	images_dir = "./images/" 
	from_labels_to_annotation(list_file,labels_dir,images_dir)
