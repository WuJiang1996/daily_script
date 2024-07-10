import os
from PIL import Image

#该脚本作用是过滤图像最大像素小于某个值的照片

directory_name = 'carcolor/'
pic_list = []

for filename in os.listdir(directory_name):
    # print(filename)
    first_path = directory_name + filename
    # print(first_path)
    for secondname in os.listdir(first_path):
        second_path = first_path + '/' + secondname
        #获取每张照片的w,h
        # print(second_path)
        img = Image.open(second_path)
        max_size = max(img.size)
        # print(max_size)
        if max_size < 80:
            pic_list.append(second_path)
            
print(len(pic_list))
for i in pic_list:
    os.remove(i)
print('done!')
