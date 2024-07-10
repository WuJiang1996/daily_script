import os
import cv2
import json
import pandas as pd
import numpy as np
from glob import glob 
from tqdm import tqdm
from IPython import embed
import base64
from labelme import utils

image_path = "./VOCdevkit/VOC2007/JPEGImages/"
image_path_1 = "./VOCdevkit/VOC2007/JPEGImages"
csv_file = "./VOCdevkit/VOC2007/Annotations/scratches.csv"
annotations = pd.read_csv(csv_file, header=0).values
#print(annotations)
total_csv_annotations = {}

for annotation in annotations:
    key = annotation[0].split('/')[-1]
    #print(key)
    value = np.array([annotation[1:]])
    #print(value)
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
    else:
        total_csv_annotations[key] = value
    #print(total_csv_annotations)

for key, value in total_csv_annotations.items():
    #print(image_path + key)
    #(height, width) = cv2.imread('C:/Users/wuj/Desktop/data_format/VOCdevkit/VOC2007/JPEGImages/r_00000.jpg').shape[:2]
    #print(height, width)
    height, width = cv2.imread(image_path + key).shape[:2]
    labelme_format = {
    "version":"3.6.16",
    "flags":{},
    "lineColor": [0, 255, 0, 128],
    "fillColor": [255, 0, 0, 128],
    "imagePath": key,
    "imageHeight": height,
    "imageWidth": width
    }
    with open(image_path + key, "rb") as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    #img = utils.img_b64_to_arr(imageData)
    labelme_format["imageData"] = imageData
    shapes = []
    for shape in value:
        label = shape[-1]
        s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
        points = [
            [shape[0], shape[1]],
            [shape[2], shape[3]]
            ]
        s["points"] = points
        shapes.append(s)
    labelme_format["shapes"] = shapes
    # with open("%s/%s" % ('./VOCdevkit/VOC2007/JPEGImages', key.replace(".jpg", ".json")), "w") as f:
    #     print("1111")
    json.dump(labelme_format, open("%s/%s" % (image_path_1, key.replace(".jpg", ".json")), "w"), ensure_ascii = False, indent = 2)