import cv2
import os
import random

#import imagesize
from pathlib import Path

#该脚本作用是切图，根据labelimg打标签出来的txt文件中的框信息，把某个文件夹下面的框还原到原图并从原图切图出来
#该脚本作用是切图，labelimg标注出来的yolo格式（xywh，中心点，w,h）坐标，转换成xyxy（左上角右下角）坐标

directory_name = '/data/windows/罐装车分类/数据/wj/'

txt_directory_name = directory_name + 'txt/'

#检验是否含有中文字符
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

for filename in os.listdir(txt_directory_name):
    #print('filename:',filename)
    if '.txt' in filename:
        txt_file_path = txt_directory_name + filename
        #print('txt_file_path:', txt_file_path)
        print('txt_file_path:',txt_file_path)
        with open(txt_file_path, 'r') as f:
            print('txt_file_path:',txt_file_path)
            data = f.readlines()
            #print(len(data))
            j = 0
            for i in data:
                list = i.split(' ')
                print(list[0])
                print(type(list[0]))
                if is_contains_chinese(list[0]):
                    continue
                elif list[0] == 'car\n':
                    continue
                elif int(list[0]) == 3: 
                    print(list[0])  
                    pic_path = directory_name + 'images/' +  filename.split('.')[0] + '.jpg'
                    #pic_path = Path(pic_path)
                    #if pic_path.is_file():
                    if os.path.exists(pic_path):
                        #width, heigth = imagesize.get(pic_path)

                        im = cv2.imread(pic_path)
                        if im is not None:
                            #print(im.shape)
                            print(type(im.shape[0]))
                            if im.shape[0] != None:
                                heigth = im.shape[0]
                                width = im.shape[1]

                                x_center = int(float(list[1]) * float(width))
                                y_center = int(float(list[2]) * float(heigth))
                                w = int(float(list[3]) * float(width))
                                h = int(float(list[4]) * float(heigth))
            
                                x1 = int(x_center - w /2)
                                y1 = int(y_center - h/2)
                                x2 = int(x_center + w /2)
                                y2 = int(y_center + h/2)

                                #im = cv2.imread(pic_path)
                                #print(im.shape)
                                cropped = im[y1:y2, x1:x2]

                                outPutDirName = './crop/' 
                                if not os.path.exists(outPutDirName):
                                    os.makedirs(outPutDirName)
                                
                                j += 1
                                outputimage_name = filename.split('.')[0] + '_' + str(j) + '.jpg'
                                print(outPutDirName + outputimage_name)
                                outputimage_name_back = 'outputimage_name'
                                try:
                                    if cropped is not None:
                                        cv2.imwrite(outPutDirName + outputimage_name, cropped)
                                except:
                                    continue
                                #print("done!")
            print('txt_file_path:',txt_file_path)
            os.remove(txt_file_path)
