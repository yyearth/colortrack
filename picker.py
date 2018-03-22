#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: youyang time:2018/3/14 20:30 

import numpy as np
import cv2

'''
click on the pixel will print the RGB color of pixel.
'''

img = cv2.imread('squ.jpg')


def pick(event, x, y, flags, param):
    # print(event, x, y, flags)
    if event == cv2.EVENT_LBUTTONUP:
        print((x, y), 'BGR color:', img[y, x])


cv2.namedWindow('img')
cv2.setMouseCallback('img', pick)

if img is not None:
    cv2.imshow('img', img)
    cv2.waitKey()
