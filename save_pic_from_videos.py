# 视频分解为单帧照片并保存到指定文件夹
# 1 load 2 info 3 parse 4 imshow imwrite
import cv2
import os

directory_name = 'mp4/'

for filename in os.listdir(directory_name):
    print('filename:', filename)
    video_path = directory_name + filename
    cap = cv2.VideoCapture(video_path)
    isOpened = cap.isOpened # 判断是否打开
    print(isOpened)

    # 获取信息 宽高
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('总帧数:',n_frame) # 整个视频的总帧数
    fps = cap.get(cv2.CAP_PROP_FPS) # 帧率 每秒展示多少张图片
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # w
    #print('w:',width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # h
    print('帧数、宽度、高度分别为:',fps,width,height) # 帧率 宽 高

    i = 0 # 记录读取多少帧
    frameFrequency = 100 # 每frameFrequency保存一张图片
    while(isOpened):
        # 结束标志是否读取到最后一帧
        if i == n_frame:
            break
        else:
            i = i+1
        (flag,frame) = cap.read() # read方法 读取每一张 flag是否读取成功 frame 读取内容
        fileName = str(filename.split('_')[2].split('.')[0]) + '_' + str(i)+'.jpg' # 名字累加
        # True表示读取成功 进行·写入
        # if 判断需要有冒号
        #if flag == True:
        outPutDirName = './car1/' # 设置保存路径
        # 如果文件目录不存在则创建目录 
        if not os.path.exists(outPutDirName):
            os.makedirs(outPutDirName)
        if i % frameFrequency == 0:
            print(fileName)
            try:
                cv2.imwrite(outPutDirName+fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])# 质量控制 100最高
            except:
                print("save photo failed!")
    cap.release()
    # os.remove(video_path)

