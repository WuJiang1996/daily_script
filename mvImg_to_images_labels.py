#coding:utf-8
#liyuanyue_2022_03_14

import os

dir_orgin = "2022-6-15" #原始文件夹
num = -1 #每个文件夹的取图数量， 1-n , -1则全部取出。
dir_new = "images" #输出文件夹
endTypes = [".jpg",".jpeg",".png"] #图片后缀名称，可删、增或修改


def getVideoPaths(dir_,endTypes,num):
	iter_ = 0
	paths = []
	ns = [os.path.join(dir_,n) for n in os.listdir(dir_)]
	for n in ns:
		if os.path.isfile(n):
			endswithFlag = False
			for endType in endTypes:
				if(n.endswith(endType)):
					endswithFlag = True
					break
			if (endswithFlag and (num<0 or iter_<num)):
				iter_ = iter_+1
				paths.append(n)
		else:
			paths = paths + getVideoPaths(n,endTypes,num)
		#print(paths)
	return paths

paths = getVideoPaths(dir_orgin,endTypes,num)
print("len : ",len(paths))
# os.system("mkdir "+dir_new)
# os.system("echo 123456 | sudo -S chmod -Rf 777 "+dir_new)
# for p in paths:
# 	os.system("cp "+p+" "+os.path.join(dir_new,p.split("/")[-1]))

