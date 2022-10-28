import cv2
import os
import shutil

#此脚本为从reid挑选出车型颜色的数据

directory_name = '4/'
list = []
color_list = []

for filename in os.listdir(directory_name):
    file_path = directory_name + filename
    # list.append(pic_path)
    # print(filename)
    i = filename.split('-')
    # print(i)
#     print((i[0],i[2]))
#     color_list.append(i[1])
#     color_list_set = set(color_list)
# print(color_list_set)
    if i[2] == '大卡车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'big_truck'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '面包车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'business_car'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '大巴':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'bus'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '厢式货车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'van'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '罐装车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'tanker'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '大货车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'big_truck'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '轿车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'sedan'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == 'SUV':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'SUV'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '小卡车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'small_truck'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
    if i[2] == '施工车':
        for j in os.listdir(file_path):
            pic_path = file_path + '/' + j
            outPutDirgzName = 'construction_truck'
            outPutDirgzPath = directory_name + outPutDirgzName
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            newpicpath = outPutDirgzPath + '/' + j
            shutil.copy(pic_path, newpicpath)
print('done!')
# print(list)
# print(len(list))



