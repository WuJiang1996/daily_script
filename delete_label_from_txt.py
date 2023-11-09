import os

dir = './labels/'
if not os.path.exists(dir):
    os.makedirs(dir)

for filename in os.listdir(dir):
    # print(filename)
    label_list = []
    txt_path = dir + filename
    if filename.endswith('txt'):
        #print(txt_path)
        with open(txt_path) as f:
            context = f.readlines()
            length = len(context)
            if length > 0:
                for i in range(length):
                    #print(int(context[i][0]))
                    if int(context[i][0]) != 5:
                        label_list.append(i)
                    # if int(context[i][0]) == 5:
                    #     print("label is 5!!!!!!!!!")
                    #     print(txt_path)
                        # del context[i]

        #print("label_list:", label_list)
        with open(txt_path) as f:
            outPutDirgzName = 'labels1'
            outPutDirgzPath = outPutDirgzName 
            if not os.path.exists(outPutDirgzPath):
                os.makedirs(outPutDirgzPath)
            file_new_path = outPutDirgzName + '/'+filename 
            #print('file_new_path:', file_new_path)
            file_new = open(file_new_path,'w')
            for j in label_list:
                file_new.writelines(context[j]) # 将删除行后的数据写入文件
            file_new.close()
print('done!')
