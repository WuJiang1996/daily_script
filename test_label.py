import os
import shutil
dir = './labels/'

for filename in os.listdir(dir):
    # print(filename)
    txt_path = dir + filename
    with open(txt_path) as f:
        context = f.readlines()
        # print(type(context))
        for i in context:
            if int(i[0]) == 6 or int(i[0]) == 6:
                print('yes!')
                print(txt_path)