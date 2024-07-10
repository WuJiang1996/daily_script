import os 
import cv2
import numpy as np


def get_illum_mean_and_std(img):
	is_gray = img.ndim == 2 or img.shape[1] == 1
	if is_gray:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	illum = hsv[..., 2] / 255.
	return np.mean(illum), np.std(illum)

def gamma_transform(img, gamma):
    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

dir_path = './img'
for filename in os.listdir(dir_path):
    print(filename)
    img_path = dir_path + '/' + filename
    img = cv2.imread(img_path)
    img_trans = gamma_transform(img, 1.5)
    new_name = dir_path + '/' + filename.split('.')[0] + 'change_color' + '.jpg'
    cv2.imwrite(new_name, img_trans)
    print('done!')
