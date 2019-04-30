# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:10:25 2019

@author: hasee
"""

import cv2 as cv

img = cv.imread("E:\py/test.jpg")

cv.namedWindow("Image")
cv.imshow("Image",img)
cv.waitKey(0)

cv.destroyAllWindows()