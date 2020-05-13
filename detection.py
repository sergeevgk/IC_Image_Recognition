import cv2 as cv
import numpy as np
from skimage.morphology import *
from skimage import color
from utility import filter_white_pixels
import matplotlib.pyplot as plt


def make_canny_edges(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    canny_edges = cv.Canny(gray_image, 0, 0)
    mask = canny_edges != 0
    colored_canny_edges = image * (mask[:, :, None].astype(image.dtype))
    return canny_edges, colored_canny_edges


def draw_canny_edges(image, ce, cce):
    cv.imshow("original", image)
    cv.imshow("edges", ce)
    cv.imshow("color edges", cce)
    cv.waitKey()


def detect_hottest_parts(image):
    fig, ax = plt.subplots(1, 3, figsize=(30, 20))
    gray_im = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ax.flatten()[0].imshow(gray_im)
    ret, thresh1 = cv.threshold(gray_im, 130, 125, cv.THRESH_BINARY)
    thresh1 = binary_closing(thresh1, selem=np.ones((3, 3)))
    ax.flatten()[1].imshow(thresh1)
    thresh1 = binary_erosion(thresh1, selem=np.ones((5, 5)))
    from scipy.ndimage import binary_fill_holes
    thresh1 = 1 - thresh1
    thresh1 = binary_fill_holes(thresh1, structure=np.ones((2,2)))
    ax.flatten()[2].imshow(thresh1)
    # plt.show()
    dwg = image.copy()
    dwg = dwg * (thresh1[:, :, None].astype(dwg.dtype))
    t = cv.inRange(dwg, 0, 125)

    t = cv.morphologyEx(t, cv.MORPH_GRADIENT, np.ones((2, 2)))

    # detect_lines(gray_im, t)
    contours, hierarchy = cv.findContours(t, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for c in contours:
        # epsilon = 0.1 * cv.arcLength(c, True)
        # approx = cv.approxPolyDP(c, epsilon, True)
        # cv.drawContours(gray_im, [approx], -1, color=(0, 0, 255), thickness=2)
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(gray_im, [box], 0, (0, 0, 255), 2)
    cv.imshow("i",gray_im)
    cv.waitKey()
    return contours, hierarchy


def detection_alg(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    canny_edges, colored_canny_edges = make_canny_edges(image)
    draw_canny_edges(image,canny_edges, colored_canny_edges)
    return image



def detect_lines(image, t):
    lines = cv.HoughLinesP(t, 0.5, np.pi / 360, 10, minLineLength=5, maxLineGap=5)
    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        theta = np.arctan(-1 / k)
        rho = b * np.sin(theta)
        line_params.append((rho, theta, x1, y1, x2, y2))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        plt.imshow(image, cmap='gray')
        plt.show()
    line_params = sorted(line_params, key=lambda x: (x[0], x[1]))

    lines_unique = []
    merged_line = []

    for i in range(len(line_params) - 1):
        rho1 = line_params[i][0]
        theta1 = line_params[i][1]
        rho2 = line_params[i + 1][0]
        theta2 = line_params[i + 1][1]
        x11 = line_params[i][2]
        y11 = line_params[i][3]
        x12 = line_params[i][4]
        y12 = line_params[i][5]
        x21 = line_params[i + 1][2]
        y21 = line_params[i + 1][3]
        x22 = line_params[i + 1][4]
        y22 = line_params[i + 1][5]
        if len(merged_line) == 0:
            merged_line = [x11, y11, x12, y12]

        if is_params_equals(rho1, theta1, rho2, theta2):
            x1m, y1m, x2m, y2m = merged_line
            x_min = np.min([x11, x12, x21, x22, x1m, x2m])
            x_max = np.max([x11, x12, x21, x22, x1m, x2m])
            y_min = np.min([y11, y12, y21, y22, y1m, y2m])
            y_max = np.max([y11, y12, y21, y22, y1m, y2m])
            if theta1 > 0:
                merged_line = [x_max, y_min, x_min, y_max]
            else:
                merged_line = [x_max, y_max, x_min, y_min]
        else:
            lines_unique.append(merged_line)
            merged_line = []

    if len(merged_line) != 0:
        lines_unique.append(merged_line)
    else:
        lines_unique.append(list(line_params[-1][2:6]))



def is_params_equals(rho1, theta1, rho2, theta2):
    rho_err = 3
    theta_err = 0.03

    if np.abs(theta1 - theta2) < theta_err and np.abs(rho1 - rho2) < rho_err:
        return True
    return False

