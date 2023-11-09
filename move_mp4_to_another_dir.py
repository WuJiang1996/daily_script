import os, random, shutil
mp4_list = []

def moveFile(label_dir):
    for dirname in os.listdir(label_dir):    #取图片的原始路径
        print("dirname",dirname)
        print(type(dirname))
        if (dirname != "temp") and (dirname != "copy") and (not dirname.endswith(".mp4")) and (not dirname.startswith("System")):
            print("1111")
            print("dirname",dirname)
            if int(dirname) >= 20231001 and int(dirname) <= 20231031:
                sec_path = label_dir + dirname + '/'
                for imglabel_path in os.listdir(sec_path):
                    # print(type(imglabel_path))
                    if imglabel_path.endswith('_bak.mp4'):
                        img_path = sec_path + imglabel_path
                        # print(img_path)
                        mp4_list.append(img_path)

    for name in mp4_list:
        img_name = name.split('/')[-1]
        shutil.copy(name, mp4Dir+img_name)

if __name__ == '__main__':
    label_dir = '/data/opt/tomcat/webapps/eventvideo/'
    mp4Dir = './20231030/'    #移动到新的文件夹路径
    if not os.path.exists(mp4Dir):
        os.makedirs(mp4Dir)
    moveFile(label_dir)
    print("mp4_list:",mp4_list)
    print(len(mp4_list))
    print("done!!!")
