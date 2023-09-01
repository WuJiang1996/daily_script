import os
from os import path
#该脚本作用是通过ffpmeg把h264文件转换成mp4文件

wdr = path.normpath(r'20230526/')
videoList = os.listdir(wdr)
#获取文件夹下所有文件列表

#ffmpegCmd = 'ffmpeg -i {} -vcodec libx264 -vf scale=720:-2 -threads 4 {}_conv.mp4'

# 11.使用ffpmpeg把MP4转为h264
# ffmpeg -i input.mp4 -codec copy -bsf: h264_mp4toannexb -f h264 output.h264
# 12.使用ffpmpeg把h264转为MP4
# ffmpeg -f h264 -i source.h264 -vcodec copy out.mp4

# ffmpegCmd = 'ffmpeg -f h264 -i {} -vcodec copy {}_conv.mp4'
ffmpegCmd = 'ffmpeg -i {} -codec copy -bsf: h264_mp4toannexb -f h264 {}.h264'

#设置ffmpeg命令模板
cmd = f'cd "{wdr}"\n{path.splitdrive(wdr)[0]}\npause\n'
#第1步，进入目标文件夹

def comprehensionCmd(e):
#手写一个小函数方便后面更新命令
    root,ext = path.splitext(e)
    # print('e:',e)
    # print('root:',root)
    return ffmpegCmd.format(e,root)

videoList = [comprehensionCmd(e) for e in videoList if not('conv' in e)]
#第3和第4步

cmd += '\n'.join(videoList)
# 将各个ffmpeg命令一行放一个

cmd += '\npause'

output = open('videoConv.bat','w')
output.write(cmd)
output.close()
#命令写入文件