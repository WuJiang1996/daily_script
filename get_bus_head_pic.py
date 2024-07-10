import cv2
import os
import shutil

#此为truck的车型分类数据处理脚本

directory_name = 'groups-car/'

for filename in os.listdir(directory_name):
    # print(filename)
    secondpath = directory_name + filename
    # print(secondpath)
    for secondname in os.listdir(secondpath):
        # print(secondname)
        if "车头" in secondname:
            thirdpath = secondpath + '/' + secondname
            # print(thirdpath)
            for thirdname in os.listdir(thirdpath):
                pic_path = thirdpath + '/' + thirdname
                outPutDirgzName = 'bus'
                outPutDirgzPath = directory_name + outPutDirgzName
                if not os.path.exists(outPutDirgzPath):
                    os.makedirs(outPutDirgzPath)
                newpicpath = outPutDirgzPath + '/' + thirdname
                shutil.copy(pic_path, newpicpath)

