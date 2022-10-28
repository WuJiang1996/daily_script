import os
import shutil
dir = './labels/'

for filename in os.listdir(dir):
    # print(filename)
    txt_path = dir + filename
    with open(txt_path) as f:
        context = f.readlines()
        length = len(context)
        for i in range(length):
            # print(int(context[i][0]))
            if int(context[i][0]) == 5:
                del context[i]

    with open(txt_path) as f:
        outPutDirgzName = 'new_label'
        outPutDirgzPath = outPutDirgzName
        if not os.path.exists(outPutDirgzPath):
            os.makedirs(outPutDirgzPath)
        file_new_path = outPutDirgzName + filename 
        file_new = open(file_new_path,'w')
        file_new.writelines(context[i]) # 将删除行后的数据写入文件
        file_new.close()
