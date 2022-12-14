##把一个文件下面部分数据（随机选取）移动到另一个文件夹下面
# import os, random, shutil
# def moveFile(fileDir):
#         pathDir = os.listdir(fileDir)    #取图片的原始路径
#         filenumber=len(pathDir)
#         rate=0.01    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#         picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
#         sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
#         print (sample)
#         for name in sample:
#                 shutil.move(fileDir+name, tarDir+name)
#         return

# if __name__ == '__main__':
# 	fileDir = "./older/"    #源图片文件夹路径
# 	tarDir = './newer/'    #移动到新的文件夹路径
# 	moveFile(fileDir)


import os, random, shutil
img_list = []

def moveFile(label_dir):
        for imgname in os.listdir(label_dir):    #取图片的原始路径
            imgname = imgname.split('.')[0] + '.jpg'
            img_list.append(imgname)
    
        for name in img_list:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
    label_dir = './labels/'
    fileDir = "./"    #源图片文件夹路径
    tarDir = './newer/'    #移动到新的文件夹路径
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    moveFile(label_dir)