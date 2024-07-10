#批量修改文件名
#批量修改图片文件名
import os
import re
import sys
import time
from datetime import datetime
# print(datetime.now().strftime('%Y%m%d%H%M%S'))

directory_name = 'images/'
output_name = 'images/'

def renameall():
    fileList = os.listdir(directory_name)       #待修改文件夹
    print("修改前："+str(fileList))     #输出文件夹中包含的文件
    currentpath = os.getcwd()       #得到进程当前工作目录
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    os.chdir(output_name)        #将当前工作目录修改为待修改文件夹的位置
    num=20230620111701       #名称变量
    for fileName in fileList:       #遍历文件夹中所有文件
        pat=".+\.(jpg)"      #匹配文件名正则表达式
        pattern = re.findall(pat,fileName)      #进行匹配
        # os.rename(fileName,(str(num)+'.'+pattern[0]))       #文件重新命名
        # os.rename(fileName,(datetime.now().strftime('%Y%m%d%H%M%S')+ str(num) + '.'+pattern[0]))       #文件重新命名
        os.rename(fileName,(str(num) + '.'+pattern[0]))       #文件重新命名
        num = num+1     #改变编号，继续下一项
    print("---------------------------------------------------")
    os.chdir(currentpath)       #改回程序运行前的工作目录
    sys.stdin.flush()       #刷新
    print("修改后："+str(os.listdir(output_name)))       #输出修改后文件夹中包含的文件

renameall()