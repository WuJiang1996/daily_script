"""
visual bounding boxes in image
meishuang 2018/09/03
"""

import os
import numpy as np
import io
import PIL
from PIL import Image, ImageDraw

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def txt2boxes(image_path, annotation_path):
    f = open(annotation_path, 'r')
    dataSet = []
    kkk = 0
    for line in f:
        infos = line.split(" ")
        length = len(infos)
        image_name_path = infos[0]
        img_data = Image.open(image_name_path.replace("\n",""))
           # img.show()
            #img_data = np.array(img)
            
        #img_ori_size = img_data.shape # height, width
        #if len(img_data.shape) == 2:
        #    img_data = np.expand_dims(img_data, 2)
         #   img_data = np.tile(img_data, [1, 1, 3])

        image_boxes = []
        for i in range(1, length):
            image_boxes_single = infos[i].split(",")
            image_boxes.append(image_boxes_single)
        if( len(image_boxes) < 1):
            print(image_name_path.replace("\n",""))
            continue
        if img_data is None:
            continue
        # xmin, ymin, xmax, ymax, label of a Nx5 array
        image_boxes = np.array(image_boxes)
        image_boxes_use = image_boxes[:, :5]
        box_classes = image_boxes[:, 4]
        #image_boxes_use = image_boxes[:, [1, 0, 3, 2, 4]][:, :4]
        #box_classes = image_boxes[:, [1, 0, 3, 2, 4]][:, 4]
		
		
        draw = ImageDraw.Draw(img_data)
        for num in range(box_classes.shape[0]):
            left = int(image_boxes_use[num,0])
            top = int(image_boxes_use[num,1])
            right = int(image_boxes_use[num,2])
            bottom = int(image_boxes_use[num,3])
            class_ = int(image_boxes_use[num,4])
            #[left, top, right, bottom] = image_boxes_use[num]
            #[left, top, right, bottom] = [15,8,40,49]#image_boxes_use[num]
            #print 	(left, top, right, bottom)
            #for thick in range(min((bottom-top)//2,(right-left)//2)+1):
            for thick in range(2):
                #draw.rectangle([str(left + thick）, str(top + thick）, str(right - thick）, str(bottom - thick）], 'black', 'red'), fill = (255,255,0))
                if(class_ == 0):
                    #draw.rectangle(xy=(left + thick, top + thick, right - thick, bottom - thick), fill=(128,128,128), outline=(128,128,128))
                    draw.rectangle((left + thick, top + thick, right - thick, bottom - thick), outline = (255,255,255))
                if(class_ == 1):
                    draw.rectangle((left + thick, top + thick, right - thick, bottom - thick), outline = (255,0,255))
                if(class_ == 2):
                    draw.rectangle((left + thick, top + thick, right - thick, bottom - thick), outline = (255,255,0))
                if(class_ == 3):
                    draw.rectangle((left + thick, top + thick, right - thick, bottom - thick), outline = (0,255,255))
                if(class_ == 4):
                    draw.rectangle((left + thick, top + thick, right - thick, bottom - thick), outline = (0,0,0))
                    #outline=self.colors[c])
                if(class_ == 5):
                    draw.rectangle((left + thick, top + thick, right - thick, bottom - thick), outline = (0,0,255))
                    #outline=self.colors[c])
        del draw
        #img_draw = draw_boxes(img_data, image_boxes_use, box_classes, classes)
        #img_draw.show()
        #img_data.save('./result/' + str(kkk) + '.png')
        os.system("mkdir results")
        if(image_name_path.replace("\n","").replace("images","results") != image_name_path):
                img_data.save(image_name_path.replace("\n","").replace("images","results"))
        kkk = kkk + 1
    f.close()
    #return result

if __name__ == '__main__':
    image_path = ''
    annotation_path = './annotation2.txt'
    txt2boxes(image_path, annotation_path)
