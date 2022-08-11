import argparse
import time
from pathlib import Path

import sys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import cv2
import torch
import torch.nn as nn
from numpy import random
import numpy as np
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device


input_size = [640,640] 




weight_path = './yolov5m.pt' 


def xywh2xyxy(x):
    x0 = x[..., 0:1]
    x1 = x[..., 1:2]
    x2 = x[..., 2:3]
    x3 = x[..., 3:4]

    y0 = x0 - x2 / 2
    y1 = x1 - x3 / 2
    y2 = x0 + x2 / 2
    y3 = x1 + x3 / 2
    y = torch.cat((y0,y1,y2,y3),2)
    return y


def nms(detections):
    boxes = detections[..., :4]
    boxes = xywh2xyxy(boxes)
    box_confidence = detections[..., 4:5]
    box_class_probs = detections[..., 5:]
    scores = box_confidence * box_class_probs
    #scores = box_class_probs
    boxes = boxes.unsqueeze(2)
    return boxes,scores


class YoloV5Detector(torch.nn.Module):
    def __init__(self, weights, img_size=(640, 640), device=torch.device('cpu')):
        super(YoloV5Detector, self).__init__()
        self.device = device
        self.load_model(weights, img_size)
    
    def load_model(self, weights, img_size):
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        # Checks
        self.stride = self.model.stride
        gs = int(max(self.stride))  # grid size (max stride)
        self.img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # Update model
        for k, m in self.model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)
    
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x /= 255.0
        pred = self.model(x, False)[0]
        boxes, scores = nms(pred)
        print("#"*50,boxes.size())
        print("#"*50,scores.size())
        return boxes, scores
        # pred = self.model(x, False)
        # return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=weight_path, help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=input_size, help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes',default=False)
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    set_logging()
    t = time.time()
    device = select_device(opt.device)
    model = YoloV5Detector(opt.weights, tuple(opt.img_size), device)
    #model = YoloV5Detector(opt.weights, (96,96), device)
    stride = model.stride
    print(model.names)

    # Input
    #img = torch.zeros(opt.batch_size, 3, *model.img_size).to(device) 

    img = torch.zeros(opt.batch_size,  *model.img_size, 3).to(device) 
    print("#"*20,img.size())
    img = cv2.imread("1.jpg")
    #img = cv2.imread("tmp.jpg")[:, :, ::-1]
    print("#"*20,img.shape)
    
    img = cv2.resize(img,(640,640))
    img = torch.from_numpy(img.copy()).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    img = img.float()
    print("#"*20,img.size())
    
    
    
    boxes, scores = model(img)
    
    for i in range(25200):
        if(scores[0,i,1]>0.1):
            print(i,scores[0,i,1])

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['input'],
                          output_names=['boxes', 'scores'],
                          dynamic_axes={'input': {0: 'batch'},  # size(1,3,640,640)
                                        'boxes': {0: 'batch'},
                                        'scores': {0: 'batch'}} if opt.dynamic else None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        from onnxsim import simplify

        if opt.dynamic:
            model_simp, check = simplify(onnx_model, dynamic_input_shape=True, input_shapes={'input': [1, input_size[0], input_size[1], 3]},check_n=0)
            # sys.exit(0)
        else:
            model_simp, check = simplify(onnx_model,check_n=0)
        assert check, "Simplified ONNX model could not be validated"

        onnx.save(model_simp, f)

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)
