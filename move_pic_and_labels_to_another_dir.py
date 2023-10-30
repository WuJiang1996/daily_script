##把一个文件下面部分数据（随机选取）移动到另一个文件夹下面
# import os, random, shutil
# def moveFile(fileDir):
#         pathDir = os.listdir(fileDir)    #取图片的原始路径
#         filenumber=len(pathDir)
#         rate=0.1   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#         picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
#         sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
#         print (sample)
#         for name in sample:
#                 shutil.move(fileDir+name, tarDir+name)
#         return

# if __name__ == '__main__':
# 	fileDir = "./car2/"    #源图片文件夹路径
# 	tarDir = './car4/'    #移动到新的文件夹路径
# 	moveFile(fileDir)

import os, random, shutil
img_list = []
label_list = []

def moveFile(label_dir):
        for dirname in os.listdir(label_dir):    #取图片的原始路径
            sec_path = label_dir + dirname + '/'
            for imglabel_path in os.listdir(sec_path):
                # print(type(imglabel_path))
                if imglabel_path.endswith('.jpg'):
                    img_path = sec_path + imglabel_path
                    # print(img_path)
                    img_list.append(img_path)
                if imglabel_path.endswith('txt'):
                    label_path = sec_path + imglabel_path
                    # print(label_path)
                    label_list.append(label_path)

        for name in img_list:
            img_name = name.split('/')[-1]
            shutil.copy(name, imgDir+img_name)
        for name in label_list:
            label_name = name.split('/')[-1]
            shutil.copy(name,labeldir+label_name)

if __name__ == '__main__':
    label_dir = './2023-10-18-广西都巴路误报-已完成/'
    imgDir = './images/'    #移动到新的文件夹路径
    labeldir = './labels/' 
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
    if not os.path.exists(labeldir):
        os.makedirs(labeldir)
    moveFile(label_dir)
    print("done!!!")
    