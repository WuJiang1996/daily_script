import cv2
import os

directory_name = 'pic/'
directory_name1 = 'new_txt/'

for filename in os.listdir(directory_name):
    print('filename:', filename)
    txt_path = directory_name + filename

    file = open(txt_path)    
    lines = file.readlines()
    print("len1:",len(lines))
    # print(type(lines[0]))

    list = []
    j = 0
    for i in lines:
        i = i.strip()
        # print(type(i[0]))
        # print(i[0])
        if int(i[0]) == 5:
            list.append(j)

        j += 1
    print(list)

    len_list = len(list)
    for i in range(len_list):
        # print(i)
        del lines[list[i] - i]
        # print('done!')
    # print(lines)
    print("len2:",len(lines))

    file.close()


    new_txt_path = directory_name1 + filename
    new_txt = open(new_txt_path,'w')
    new_txt.writelines(lines) # 将删除行后的数据写入文件
    new_txt.close()
