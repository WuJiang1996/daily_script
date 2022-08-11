import os 
import cv2
#该脚本作用是删除多余的txt文件

array_of_txt = [] 
array_of_jpg = []
directory_name = 'images/'

for filename in os.listdir(directory_name):
    # print('filename:',filename)
    if filename.endswith('txt'):
        array_of_txt.append(filename)
print(len(array_of_txt))
#print(array_of_img)

for filename in os.listdir(directory_name):
    # print('filename:',filename)
    if filename.endswith('jpg'):
        array_of_jpg.append(filename)
print(len(array_of_jpg))

        
for name in array_of_txt:
    #print('name:',name)
    first_name = name.split('.')[0]
    #print('first_name:',first_name)
    jpg_path = first_name + '.jpg'
    txt_path = 'images/' + name
    if jpg_path not in array_of_jpg:
        print("not in!")
        #os.remove(txt_path)