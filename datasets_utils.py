# camera-ready

import os
import cv2
import math
import random
import numpy as np

# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import random
import numpy as np

def gussian_blur(img, blur_prob):
    """
    input: img readed by cv2
           the probability to blur img
    output: blur_img
    """
    if random.random() < blur_prob:
        size = 3
        kernel_size = (size, size)
        sigma = random.uniform(1, 2)
        blur_img = cv2.GaussianBlur(img, kernel_size, sigma)
        return blur_img
    else:
        return img

def distort_image(img, hue, sat, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype('float32')
    hue_c = hsv[:,:,0]
    hue_c += hue
    hue_c[hue_c > 180.] -= 180.
    hue_c[hue_c < 0.] += 180.
    sat_c = hsv[:,:,1]
    sat_c *= sat
    sat_c[sat_c > 255.] = 255.
    val_c = hsv[:,:,2]
    val_c *= val
    val_c[val_c > 255.] = 255.
    # jitter
    bgr = np.uint8(hsv)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_HSV2BGR)
    return bgr

def random_distort_image(img, distort_prob, hue=1, sat=0.85, val=0.85):
    if random.random() < distort_prob:
        dhue = np.random.uniform(-hue, hue)
        dsat = np.random.uniform(sat, 1./sat)
        dval = np.random.uniform(val, 1./val)
        distort_img =  distort_image(img, dhue, dsat, dval) 
        return distort_img
    else:
        return img

def random_wave(img, wave_prob, mu=0, sigma=10): #高斯随机噪声
    if random.random() < wave_prob:

        h, w, _ = img.shape
        img = np.float32(img)
        img[:,:,0] += np.random.normal(mu, sigma, h*w).reshape(h, w)
        img[:,:,1] += np.random.normal(mu, sigma, h*w).reshape(h, w)
        img[:,:,2] += np.random.normal(mu, sigma, h*w).reshape(h, w)
        img[img > 255.] = 255.
        img[img < 0.] = 0.
        return np.uint8(img)
    else:
        return img


def random_crop(img, crop_prob):
    if random.random() < crop_prob:
        h, w, _ = img.shape
        crop_h = random.randint(int(h*0.93), h-1)
        crop_w = random.randint(int(w*0.93), w-1)
        h_off = random.randint(0, h - crop_h)
        w_off = random.randint(0, w - crop_w)
        return img[h_off:h_off+crop_h, w_off:w_off+crop_w]
    else:
        return img
        
if __name__ == "__main__":
    for i in range(1):
        img = cv2.imread("1.jpg")
        #img = random_crop(img, 1)
        #img = random_wave(img, 1)
        img = random_distort_image(img, 1)
        #img = gussian_blur(img, 1)
        cv2.imwrite("res/"+str(i)+".jpg",img)
        cv2.imwrite("0.jpg",img)

