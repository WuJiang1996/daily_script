import os 
import cv2
#该脚本作用是给jpg照片没有对应txt的创建txt文件

array_of_img = [] 
directory_name = '2022-4-20_yanhuo_wurenji_smoke/'

for filename in os.listdir(directory_name):
    # print('filename:',filename)
    array_of_img.append(filename)

#print('array_of_img:', array_of_img)

for filename in os.listdir(directory_name):
    # print('filename:',filename)
    filename = filename.replace('jpg', 'txt')
    # print(filename)
    full_path = directory_name + filename
    if filename not in array_of_img: 
        file = open(full_path, 'w') 
        file.write('') 
        file.close() 

