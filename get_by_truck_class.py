import cv2
import os
import shutil

#此为truck的车型分类数据处理脚本

directory_name = 'groups-truck/'

for filename in os.listdir(directory_name):
    if "已完成" in filename:
        secondpath = directory_name + filename
        #print(secondpath)
        for secondname in os.listdir(secondpath):
            #print(secondname)
            if secondname == "车类别":
                thirdpath = secondpath + '/' + secondname
                #print(thirdpath)
                for thirdname in os.listdir(thirdpath):
                    #print(thirdname)
                    if thirdname == '罐装车':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'tanker'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    #如果是其他类型统一放到一个文件夹
                    elif thirdname == '大卡车':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'big_trucks'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '施工车':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'construction_vehicle'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '厢式货车':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'van'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '小卡车':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'pickup_truck'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
