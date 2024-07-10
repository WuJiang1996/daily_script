import os
import numpy as np


f = open("train_list.txt")
ls = f.readlines()
print(ls)

dict_ = {}


for l in ls:
	a = l.split(" ")
	if a[1] in dict_.keys():
		dict_[a[1]].append(a[0])
	else:
		dict_[a[1]] = [a[0]]

print(dict_)


for k in dict_.keys():
	os.system("mkdir tmp/"+str(k))
	for n in dict_[k]:
		nn = n+".jpg"
		os.system("cp image/"+nn+"  tmp/"+str(k))
