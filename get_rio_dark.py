import os
import cv2

img_dir = './car_type_axel_20230306/train/images/'
label_dir = './car_type_axel_20230306/train/labels/'

for label_name in os.listdir(label_dir):
    label_txt_path = label_dir + label_name
    with open(label_txt_path, "r") as f:
        data = f.readlines()
        # print(data)
        for i in data:
            # print(int(i.split(' ')[0]))
            if int(i.split(' ')[0]) == 8 or int(i.split(' ')[0]) == 10:
                # print('yes!')
                # print(i)
                img_name = label_name.split('.')[0] + '.jpg'
                img_path = img_dir + img_name
                image = cv2.imread(img_path)
                image1 = image.copy()

                W = image.shape[1]
                H = image.shape[0]
                x = float(i.split(' ')[1])
                y = float(i.split(' ')[2])
                w = float(i.split(' ')[3])
                h = float(i.split(' ')[4])

                xmin=int((x-w/2)*W)
                ymin=int((y-h/2)*H)
                xmax=int((x+w/2)*W)
                ymax=int((y+h/2)*H)
                image1[ymin:ymax,xmin:xmax] = 0
                save_path = './images/' + img_name
                cv2.imwrite(save_path, image1)