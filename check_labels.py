import os 
import cv2
#该脚本作用是找到标签有问题的照片

array_of_img = [] 
directory_name = 'labels/'

for filename in os.listdir(directory_name):
    #print('filename:',filename)
    if filename.endswith('txt'):
        array_of_img.append(filename)
# print(array_of_img)

list = []

for name in array_of_img:
    txt_path = directory_name + name
    f=open(txt_path, encoding='utf-8')
    for line in f:
        # print(type(line[0]))
        l = line.split(' ')[:-1]
        # print(l)
        # print(txt_path)
        # if int(l[0]) >= 1:
        #     print(txt_path)
        if int(l[0]) != 0:
            print(txt_path)