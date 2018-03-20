#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: youyang time:2018/3/14 20:19 


import numpy as np
import cv2


#
# bgr = [77, 177, 35]
# color_low = np.array([bgr[0] - 10, 100, 100])
# color_up = np.array([bgr[0] + 10, 255, 255])

# frame = np.zeros([320, 640], np.uint8)

def bgr2hsv(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)


def process(img, color_l, color_u):
    img2 = cv2.GaussianBlur(img, (5, 5), 1)
    img_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, color_l, color_u)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)

    mask, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt_max = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt_max)
        cv2.rectangle(img, (x, y), (x + w, y + h), (150, 255, 0), 2)

    return img


# img = cv2.imread('../squ.jpg')
#
# img = process(img, color_low,color_up)
# bgr2hsv([0, 0, 0])
# cv2.imshow('img', img)
# cv2.waitKey()

if __name__ == '__main__':

    def mouse(event, x, y, flags, param):
        # print(x,y)
        global bgr, color_up, color_low
        if event == cv2.EVENT_LBUTTONUP:
            bgr = frame[y, x]
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
            color_low = np.array([cv2.add(hsv[0][0], - 10)[0][0], 100, 100], np.uint8)
            color_up = np.array([cv2.add(hsv[0][0], 10)[0][0], 255, 255], np.uint8)
            print(bgr, hsv, color_low, color_up)


    bgr = [0, 0, 0]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
    color_low = np.array([0, 0, 0], np.uint8)
    color_up = np.array([0, 0, 0], np.uint8)
    # print(bgr, color_low, color_up)
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', mouse)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            # img = cv2.GaussianBlur(frame, (5, 5), 2)
            img = process(frame, color_low, color_up)
            cv2.imshow('img', img)

            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()
    cap.release()
