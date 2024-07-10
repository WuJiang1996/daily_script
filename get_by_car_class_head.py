import cv2
import os
import shutil

# 此为truck的车型分类数据处理脚本

directory_name = 'group-car/'

# for filename in os.listdir(directory_name):
#     # print(filename)
#     secondpath = directory_name + filename
#     #print(secondpath)
#     for secondname in os.listdir(secondpath):
#         #print(secondname)
#         if secondname == "车朝向":
#             thirdpath = secondpath + '/' + secondname
#             # print(thirdpath)
#             for thirdname in os.listdir(thirdpath):
#                 #print(thirdname)
#                 if thirdname == '车头':
#                     fourthpath = thirdpath + '/' + thirdname
#                     # print(fourthpath)
#                     for picname in os.listdir(fourthpath):
#                         #print(picname)
#                         picpath = fourthpath + '/' + picname
#                         outPutDirgzName = 'head'
#                         outPutDirgzPath = directory_name + outPutDirgzName
#                         if not os.path.exists(outPutDirgzPath):
#                             os.makedirs(outPutDirgzPath)
#                         newpicpath = outPutDirgzPath + '/' + picname
#                         shutil.copy(picpath, newpicpath)

head_path = directory_name + 'head'
pic_list = [i for i in os.listdir(head_path)]
# print(pic_list)

large_truck_path = directory_name + 'data/business_car/' 
large_truck_list = [i for i in os.listdir(large_truck_path)] 

construction_truck_path = directory_name + 'data/sedan/' 
construction_truck_path_list = [i for i in os.listdir(construction_truck_path)]

small_truck_path = directory_name + 'data/suv/' 
small_truck_list = [i for i in os.listdir(small_truck_path)]

# van_path = directory_name + 'data/van/' 
# van_list = [i for i in os.listdir(van_path)]

# tanker_path = directory_name + 'data/tanker/' 
# tanker_list = [i for i in os.listdir(tanker_path)]

for i in pic_list:
    if i in large_truck_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'large_truck'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in construction_truck_path_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'construction_vehicle'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in small_truck_list:
        picpath = head_path + '/' + i
        outPutDirgzName = 'small_truck'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    # elif i in van_list:
    #     picpath = head_path + '/' + i
    #     outPutDirgzName = 'van'
    #     outPutDirgzPath = directory_name + outPutDirgzName
    #     if not os.path.exists(outPutDirgzPath):
    #         os.makedirs(outPutDirgzPath)
    #     newpicpath = outPutDirgzPath + '/' + i
    #     shutil.copy(picpath, newpicpath)
    #     continue
    
    # elif i in tanker_list:
    #     picpath = head_path + '/' + i
    #     outPutDirgzName = 'tanker'
    #     outPutDirgzPath = directory_name + outPutDirgzName
    #     if not os.path.exists(outPutDirgzPath):
    #         os.makedirs(outPutDirgzPath)
    #     newpicpath = outPutDirgzPath + '/' + i
    #     shutil.copy(picpath, newpicpath)
    #     continue
    else:
        print("not found!")
