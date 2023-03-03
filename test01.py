import os
import shutil
dir = './test1207/'
#此脚本作用是匹配车辆关键点中txt和json的对应文件（因为之前打标是labelme和labelimg，有部分车框了没关键点，有关键点但是没车框的情况）
#把对应的车辆框和关键点的标签整理到一个txt文件夹

#1.把所有文件移动到一个文件夹下面
# import shutil

# def remove_file(old_path, new_path):
#     print(old_path)
#     print(new_path)
#     filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
#     print(filelist)
#     for directory in filelist:
#         dir_path = dir + directory
#         for filename in os.listdir(dir_path):
#             # print(filename)
#             src = os.path.join(dir_path, filename)
#             dst = os.path.join(new_path, filename)
#             print('src:', src)
#             print('dst:', dst)
#             shutil.move(src, dst)

# remove_file(r"./car_key_point", r"./car_key_point_1206")

#2.删除有json（关键点）没有txt（位置置信度）或者有txt没有json或者两者都没有的数据
list_jpg = []
for i in os.listdir(dir):
    if i.endswith('jpg'):
        list_jpg.append(i)
print('list_jpg:',len(list_jpg))
# print(list_jpg)

# all_list = []
# for j in os.listdir(dir):
#     all_list.append(j)
# print('list_jpg:',len(all_list))
# print(all_list)

# for l in list_jpg:
#     txt_name = l.split('.')[0] + '.txt'
#     print(txt_name)
#     if txt_name not in all_list:
#         pass
#     json_name = l.split('.')[0] + '.json'
#     print(json_name)

import pandas as pd
import json
import math

error_list = []

for l in list_jpg:
    txt_path = dir + l.split('.')[0] + '.txt'
    json_path = dir + l.split('.')[0] + '.json'
    img_path = dir + l
    # print(txt_path)

    #解析关键点的json文件，把所有的关键点坐标都放到字典内
    point_dict = {}
    with open(json_path,'r',encoding='UTF-8') as load_f:
        load_dict = json.load(load_f)
        # len_json = len(load_dict['shapes']) / 2
        len_json = len(load_dict['shapes'])
        # print(len_json)
        # print('img:',l)
        w = load_dict['imageWidth']
        h = load_dict['imageHeight']
        for i in range(len_json):
            # print(load_dict['shapes'][i]['points'][0])
            point_x = int(load_dict['shapes'][i]['points'][0][0])
            point_y = int(load_dict['shapes'][i]['points'][0][1])
            point_dict[i] = [point_x, point_y]
        print('point_dict:', point_dict)

        #如果满足关键点在标的车框内部，就把关键点归一化之后添加到对应的标签后面
        txtname = l.split('.')[0] + '.txt'
        savetxt = open('labels/'+txtname,'w+')
        with open(txt_path) as f:
            context = f.readlines() 
            len_txt = len(context)
            # print('context:', context)
            for i in range (len_txt):
                x_center = math.ceil(float(context[i].split(' ')[1]) * w )
                y_center = math.ceil(float(context[i].split(' ')[2]) * h)
                w_bbox = math.ceil(float(context[i].split(' ')[3]) * w)
                # print(i.split(' ')[4].split('\n'))
                h_bbox = math.ceil(float(context[i].split(' ')[4].split('\n')[0]) * h)
                # print('x_center:',x_center)

                w_min = x_center - int(w_bbox / 2) - 1
                w_max = x_center + int(w_bbox / 2) + 1
                h_min = y_center - int(h_bbox / 2) - 1
                h_max = y_center + int(h_bbox / 2) + 1
                print('bbox:', w_min,w_max,h_min,h_max)

                flag = False 
                for key, value in point_dict.items():
                    x_point, y_point = value
                    # print(x_point,y_point)
                    #判断关键点是否在bbox框内
                    if x_point>=w_min and x_point<=w_max and y_point>=h_min and y_point<=h_max:
                        flag = True  
                        #如果在框内就添加到对应的txt框标签后面
                        # print('yes!')
                        x_point = round(float(x_point / w),8)
                        y_point = round(float(y_point / h),8)
                        context[i] = context[i].split('\n')[0] + ' ' + str(x_point) + ' ' + str(y_point) + '\n'
                    #如果不在范围内，说明标的关键点可能没有标车的框，此时直接丢弃
                    else: 
                        pass
                if flag:
                    savetxt.write(context[i])
    print('done!')

            # if len_txt >1:
            #     print(l) 
            # print(len_txt)
    
    # if len_txt == len_json:
        # shutil.move(img_path, r"./car_key_point")
        # shutil.move(txt_path, r"./car_key_point")
        # shutil.move(json_path, r"./car_key_point")
        # error_list.append(l)

# print('error_list:', error_list)
# print(len(error_list))


