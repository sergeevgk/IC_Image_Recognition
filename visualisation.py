# модуль для визуализации данных
import cv2
import numpy as np
from utility import color_dict as border_colors
# 0 circle ; 1 rectangle ; 2 square
label_colors_low = {0: 'blue', 1: 'green', 2: 'green'}
label_colors_high = {0: 'blue', 1: 'green', 2: 'red'}

label_colors = label_colors_high

def visualise(image, contours, labels):
    # @param image : grayscale canvas for contours
    # @param contours : filtered result of cv2.findContours
    # @param labels : results of contour classification
    # draws fit_rect of contours on the image

    #grayscale -> rgb gray colors
    # gray_three = cv2.merge([image,image,image])

    for cnt, lbl in zip(contours, labels):
        lbl = np.argmax(lbl)
        rect = cv2.minAreaRect(cnt[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, border_colors[label_colors[lbl]], 2)
    cv2.imshow("result", image)
    cv2.waitKey()

    pass


