import os 
import cv2
#该脚本作用是修改txt文件内的标签id

array_of_img = [] 
directory_name = 'txt/'

# for filename in os.listdir(directory_name):
#     # print('filename:',filename)
#     array_of_img.append(filename)

#print('array_of_img:', array_of_img)

for filename in os.listdir(directory_name):
    # print('filename:',filename)
    if filename.endswith('txt'):
        position = directory_name + filename
        savetxt = open('labels/'+filename,'w+')
        with open(position) as f:
            file = f.readlines()
            for i in file:
                i = i.split(' ')
                # print(type(i[0]))
                if int(i[0]) == 15:
                    i[0].replace('15', '1')
                    for j in i:
                        savetxt.write(j + '\t')
