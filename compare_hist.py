import cv2
import numpy as np
import time

from cv2 import COLOR_BGR2HSV

img1 = cv2.imread('C:/ICP/buterfly_1.jpg')
img1 = cv2.cvtColor(img1, COLOR_BGR2HSV)
img2 = cv2.imread('C:/ICP/buterfly_0.jpg')
img2 = cv2.cvtColor(img2, COLOR_BGR2HSV)
h = np.zeros((300, 256, 3))

bins = np.arange(256).reshape(256, 1)
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for ch, col in enumerate(color):
    hist_item1 = cv2.calcHist([img1], [ch], None, [256], [0, 255])
    hist_item2 = cv2.calcHist([img2], [ch], None, [256], [0, 255])
    cv2.normalize(hist_item1, hist_item1, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist_item2, hist_item2, 0, 255, cv2.NORM_MINMAX)
    sc = cv2.compareHist(hist_item1, hist_item2, cv2.HISTCMP_CORREL)
    print(sc)
    hist = np.int32(np.around(hist_item1))
    pts = np.column_stack((bins, hist))
    cv2.polylines(h, [pts], False, col)

h = np.flipud(h)
cv2.imwrite('C:/hist.png', h)
