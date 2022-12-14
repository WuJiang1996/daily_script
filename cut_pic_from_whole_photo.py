from wave import Wave_write
import cv2
import os
import random

#该脚本作用是切图，已知照片的xyxy坐标，从原图中切出来对应的bbox

directory_name = 'exp/'

txt_directory_name = directory_name + 'labels/'

for filename in os.listdir(txt_directory_name):
    print('filename:',filename)
    if '.txt' in filename:
        txt_file_path = txt_directory_name + filename
        print('txt_file_path:', txt_file_path)
        with open(txt_file_path, 'r') as f:
            data = f.readlines()
            #print(len(data))
            j = 0
            l = 0
            m = 0
            for i in data:
                list = i.split(' ')
                vehicle_name = int(list[0])
                if vehicle_name == 1:
                    x1 = int(list[1])
                    y1 = int(list[2])
                    x2 = int(list[3])
                    y2 = int(list[4])
                    
                    #print('filename:',filename)
                    pic_path = directory_name + 'car/' + filename.split('.')[0] + '.jpg'
                    print('pic_path:',pic_path)
                    im = cv2.imread(pic_path)
                    print(im.shape)
                    h,w,_ = im.shape
                    sss = 0.1
                    ww = x2-x1
                    hh = y2-y1
                    x1 = int(max(int(list[1])-sss*ww, 1))
                    y1 = int(max(int(list[2])-sss*hh, 1))
                    x2 = int(min(int(list[3])+sss*ww, w - 1))
                    y2 = int(min(int(list[4])+sss*hh, h - 1))

                    if(x2-x1)>80 or (y2-y1)>80:
                        cropped = im[y1:y2, x1:x2]
                        outPutDirName = './car/' 
                        if not os.path.exists(outPutDirName):
                            os.makedirs(outPutDirName)
                        
                        j += 1
                        #outputimage_name = filename.split('.')[0] + '_' + str(random.random()) + '.jpg'
                        outputimage_name = filename.split('.')[0] + '_' + str(j) + '.jpg'
                        print(outputimage_name)
                        outputimage_name_back = 'outputimage_name'
                        cv2.imwrite(outPutDirName + outputimage_name, cropped)

                elif vehicle_name == 2:
                    x1 = int(list[1])
                    y1 = int(list[2])
                    x2 = int(list[3])
                    y2 = int(list[4])
                    
                    #print('filename:',filename)
                    pic_path = directory_name + 'car/' + filename.split('.')[0] + '.jpg'
                    print('pic_path:',pic_path)
                    im = cv2.imread(pic_path)
                    print(im.shape)
                    h,w,_ = im.shape
                    sss = 0.1
                    ww = x2-x1
                    hh = y2-y1
                    x1 = int(max(int(list[1])-sss*ww, 1))
                    y1 = int(max(int(list[2])-sss*hh, 1))
                    x2 = int(min(int(list[3])+sss*ww, w - 1))
                    y2 = int(min(int(list[4])+sss*hh, h - 1))
                    if(x2-x1)>80 or (y2-y1)>80:
                        cropped = im[y1:y2, x1:x2]
                        #cv2.imshow('cropped', cropped)
                        outPutDirName = './bus/' 
                        if not os.path.exists(outPutDirName):
                            os.makedirs(outPutDirName)
                        
                        l += 1
                        #outputimage_name = filename.split('.')[0] + '_' + str(random.random()) + '.jpg'
                        outputimage_name = filename.split('.')[0] + '_' + str(l) + '.jpg'
                        print(outputimage_name)
                        outputimage_name_back = 'outputimage_name'
                        cv2.imwrite(outPutDirName + outputimage_name, cropped)

                elif vehicle_name == 3:
                    x1 = int(list[1])
                    y1 = int(list[2])
                    x2 = int(list[3])
                    y2 = int(list[4])
                    
                    #print('filename:',filename)
                    pic_path = directory_name + 'car/' + filename.split('.')[0] + '.jpg'
                    print('pic_path:',pic_path)
                    im = cv2.imread(pic_path)
                    print(im.shape)
                    h,w,_ = im.shape
                    sss = 0.1
                    ww = x2-x1
                    hh = y2-y1
                    x1 = int(max(int(list[1])-sss*ww, 1))
                    y1 = int(max(int(list[2])-sss*hh, 1))
                    x2 = int(min(int(list[3])+sss*ww, w - 1))
                    y2 = int(min(int(list[4])+sss*hh, h - 1))
                    if(x2-x1)>80 or (y2-y1)>80:
                        cropped = im[y1:y2, x1:x2]
                        #cv2.imshow('cropped', cropped)
                        outPutDirName = './truck/' 
                        if not os.path.exists(outPutDirName):
                            os.makedirs(outPutDirName)
                        
                        m += 1
                        #outputimage_name = filename.split('.')[0] + '_' + str(random.random()) + '.jpg'
                        outputimage_name = filename.split('.')[0] + '_' + str(m) + '.jpg'
                        print(outputimage_name)
                        outputimage_name_back = 'outputimage_name'
                        cv2.imwrite(outPutDirName + outputimage_name, cropped)

