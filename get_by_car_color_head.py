import os
import shutil


#此为truck的车型分类数据处理脚本

directory_name = 'group-car/'

# for filename in os.listdir(directory_name):
#     secondpath = directory_name + filename
#     # print(secondpath)
#     for secondname in os.listdir(secondpath):
#         # print(secondname)
#         if secondname == "车朝向":
#             thirdpath = secondpath + '/' + secondname
#             print(thirdpath)
    #         for thirdname in os.listdir(thirdpath):
    #             # print(thirdname)
    #             fourthpath = thirdpath + '/' + thirdname
    #             print(fourthpath)
                # for picname in os.listdir(fourthpath):
                #     #print(picname)
                #     picpath = fourthpath + '/' + picname
                #     outPutDirgzName = 'color'
                #     outPutDirgzPath = directory_name + outPutDirgzName
                #     if not os.path.exists(outPutDirgzPath):
                #         os.makedirs(outPutDirgzPath)
                #     newpicpath = outPutDirgzPath + '/' + picname
                #     shutil.copy(picpath, newpicpath)

color_path = directory_name + 'head'
pic_list = [i for i in os.listdir(color_path)]

black_path = directory_name + 'data/black/' 
black_list = [i for i in os.listdir(black_path)]

blue_path = directory_name + 'data/blue/' 
blue_list = [i for i in os.listdir(blue_path)]

dark_grey_path = directory_name + 'data/dark_grey/' 
dark_grey_list = [i for i in os.listdir(dark_grey_path)]

green_path = directory_name + 'data/green/' 
green_list = [i for i in os.listdir(green_path)]

other_path = directory_name + 'data/other/' 
other_list = [i for i in os.listdir(other_path)]

purple_path = directory_name + 'data/purple/' 
purple_list = [i for i in os.listdir(purple_path)]

red_path = directory_name + 'data/red/' 
red_list = [i for i in os.listdir(red_path)]

silver_path = directory_name + 'data/silver/' 
silver_list = [i for i in os.listdir(silver_path)]

white_path = directory_name + 'data/white/' 
white_list = [i for i in os.listdir(white_path)]

yellow_path = directory_name + 'data/yellow/' 
yellow_list = [i for i in os.listdir(yellow_path)]

for i in pic_list:
    if i in black_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'black'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in blue_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'blue'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in dark_grey_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'dark_grey'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in green_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'green'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue
    
    elif i in other_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'other'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in purple_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'purple'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in red_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'red'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in silver_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'silver'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in white_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'white'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue

    elif i in yellow_list:
        picpath = color_path + '/' + i
        outPutDirgzName = 'yellow'
        outPutDirgzPath = directory_name + outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        newpicpath = outPutDirgzPath + '/' + i
        shutil.copy(picpath, newpicpath)
        continue
    else:
        print("not found!")
