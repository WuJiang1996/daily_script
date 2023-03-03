import cv2
import os

# 读取图片

path = './car'
# img = '2_3_calib_1040.jpg'

for i in os.listdir(path):
    pic_path = path  + '/' + i 
    img = cv2.imread(pic_path, 1)
    # cv2.imshow('original', img)

    # 选择ROI
    # roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
    # x, y, w, h = roi
    # print(roi)
    x = 750
    y = 750
    w = 400
    h = 500

    # 显示ROI并保存图片
    # if roi != (0, 0, 0, 0):
    crop = img[y:y+h, x:x+w]
    # cv2.imshow('crop', crop)
    crop_path = './car_cropped/' + i
    cv2.imwrite(crop_path, crop)
    print('Saved!')

# # 退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()