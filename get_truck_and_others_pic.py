import cv2
import os
import shutil

#此为油罐车分类脚本，把文件夹下油罐车和其他truck数据分别移动到两个文件夹下面

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
                    else:
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建其他truck文件夹
                            outPutDirgzName = 'truck'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
