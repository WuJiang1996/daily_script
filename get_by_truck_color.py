import cv2
import os
import shutil

#此为truck的颜色分类数据处理脚本

directory_name = 'groups-truck/'

for filename in os.listdir(directory_name):
    if "已完成" in filename:
        secondpath = directory_name + filename
        #print(secondpath)
        for secondname in os.listdir(secondpath):
            #print(secondname)
            if secondname == "车颜色":
                thirdpath = secondpath + '/' + secondname
                #print(thirdpath)
                for thirdname in os.listdir(thirdpath):
                    #print(thirdname)
                    if thirdname == '白色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'white'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    #如果是其他类型统一放到一个文件夹
                    elif thirdname == '黑色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'black'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '红色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'red'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '黄色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'yellow'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '蓝色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'blue'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '绿色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'green'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '其他':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'other'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '深灰':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'dark_grey'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '银色（灰色）':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'silver'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
                    elif thirdname == '紫色':
                        old_file_path = thirdpath + '/' + thirdname
                        #print(old_file_path)
                        for picname in os.listdir(old_file_path):
                            #print(picname)
                            oldpicpath = old_file_path + '/' + picname
                            #print(picpath)
                            # #创建罐装车文件夹
                            outPutDirgzName = 'purple'
                            outPutDirgzPath = directory_name + outPutDirgzName
                            if not os.path.exists(outPutDirgzPath):
                                os.makedirs(outPutDirgzPath)
                            newpicpath = outPutDirgzPath + '/' + picname
                            shutil.copy(oldpicpath, newpicpath)
