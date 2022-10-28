from re import I
import cv2
import os
import shutil

from requests import head

#此为truck的车型分类数据处理脚本

directory_name = 'groups-truck/'
# j = 0

# for filename in os.listdir(directory_name):
#     if "已完成" in filename:
#         secondpath = directory_name + filename
#         #print(secondpath)
#         for secondname in os.listdir(secondpath):
#             #print(secondname)
#             if secondname == "车位置" or secondname == "车朝向":
#                 thirdpath = secondpath + '/' + secondname
#                 #print(thirdpath)
#                 for thirdname in os.listdir(thirdpath):
#                     #print(thirdname)
#                     if thirdname == '车头':
#                         fourthpath = thirdpath + '/' + thirdname
#                         print(fourthpath)
#                         for picname in os.listdir(fourthpath):
#                             #print(picname)
#                             picpath = fourthpath + '/' + picname
#                             outPutDirgzName = 'head'
#                             outPutDirgzPath = directory_name + outPutDirgzName
#                             if not os.path.exists(outPutDirgzPath):
#                                 os.makedirs(outPutDirgzPath)
#                             newpicpath = outPutDirgzPath + '/' + picname
#                             shutil.copy(picpath, newpicpath)

head_path = directory_name + 'head'
pic_list = [i for i in os.listdir(head_path)]

big_trucks_path = directory_name + 'data/big_trucks/' 
big_trucks_list = [i for i in os.listdir(big_trucks_path)]
# print(big_trucks_list)
# print(len(big_trucks_list))  
construction_vehicle_path = directory_name + 'data/construction_vehicle/' 
construction_vehicle_path_list = [i for i in os.listdir(construction_vehicle_path)]
pickup_truck_path = directory_name + 'data/pickup_truck/' 
pickup_truck_list = [i for i in os.listdir(pickup_truck_path)]
van_path = directory_name + 'data/van/' 
van_list = [i for i in os.listdir(van_path)]
tanker_path = directory_name + 'data/tanker/' 
tanker_list = [i for i in os.listdir(tanker_path)]

for i in pic_list:
    if i in big_trucks_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'big_trucks'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in construction_vehicle_path_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'construction_vehicle'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in pickup_truck_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'pickup_truck'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in van_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'van'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue
    
    elif i in tanker_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'tanker'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue
    else:
        print("not found!")
