import cv2 as cv
import numpy as np
import math
import os

color_dict = {'blue': (255, 0, 0), 'green': (0, 255, 0),
              'red': (0, 0, 255), 'yellow': (0, 255, 255),
              'white': (255, 255, 255), 'black': (0, 0, 0),
              'magenta': (255, 0, 255), 'orange': (0, 128, 255)}

#
# def calculate_median_pixel(pix_array):
#     res = np.array([0, 0, 0])
#     for pix in pix_array:
#         res = res + pix
#     res = (1 / pix_array.shape[0]) * res
#     res = map(lambda x: math.ceil(x), res)
#     return res
#
#
# def filter_white_pixels(image, ksize):
#     image_copy = image.copy()
#     h = image.shape[0]
#     w = image.shape[1]
#     for i in range(h):
#         for j in range(ksize, w - ksize):
#             if (image_copy[i, j] == np.array([255, 255, 255])).all():
#                 image_copy[i, j] = calculate_median_pixel(image_copy[i, j - ksize:j + ksize])
#     return image_copy


def scale_image(input_image_path, output_image_path, width=None, height=None):
    from PIL import Image
    original_image = Image.open(input_image_path)
    w, h = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=w, height=h))

    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')

    original_image = original_image.resize(max_size, Image.BICUBIC)
    original_image.save(output_image_path)

    scaled_image = Image.open(output_image_path)
    width, height = scaled_image.size
    print('The scaled image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))


def draw_borders_on_image(source, contours):
    image = source.copy()
    borders, hierarchy = contours
    cv.drawContours(image, borders, -1, (0, 0, 255), 3, cv.LINE_AA, hierarchy, 1)
    cv.imshow("contours", image)
    cv.waitKey()


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv.boxPoints(rect0)
    pts = np.int0(cv.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
               pts[1][0]:pts[2][0]]

    return img_crop


def getSubImage(rect, src):
    # Get center, size, and angle from rect
    r = list(rect)
    r[0] = r[0] + r[2] / 2
    r[1] = r[1] + r[3] / 2
    r[2] = r[2] + 3
    r[3] = r[3] + 3
    size = tuple(r[2:4])
    center = tuple(r[0:2])
    center, size = tuple(map(int, center)), tuple(map(int, size))
    out = cv.getRectSubPix(src, size, center)
    return out


def scale_to_nxn(im, n):
    from PIL import Image
    image = Image.fromarray(im)
    w, h = image.size
    size = []
    if w >= h:
        size = [n, round(n * h / w)]
    else:
        size = [round(n * w / h), n]
    size = tuple(size)
    image = image.resize(size, Image.BICUBIC)
    new_size = (n, n)
    new_im = Image.new("L", new_size)
    new_im.paste(image, (round((new_size[0] - size[0]) / 2),
                         round((new_size[1] - size[1]) / 2)))
    return np.array(new_im)


def save_distinct_parts(images, partition, path):
    # выделить фрагменты
    from scipy.ndimage import binary_fill_holes
    k = 0
    for cnt in partition:
        c, lvl = cnt
        rect = cv.boundingRect(c)
        cr = getSubImage(rect, images[lvl])
        cr = scale_to_nxn(cr, 32)
        t = cv.inRange(cr, 1 ,255)
        # cv.imshow("0", t)
        t = binary_fill_holes(t, structure=np.ones((5,5))).astype(int)
        t = cv.inRange(t, 1 ,255)
        # cv.imshow("1", t)
        # cv.waitKey()
        cv.imwrite(path + str(k) + ".jpg", 255 - t)
        k = k + 1
    pass


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx ** 2 + dy ** 2)


def get_length(cnt):
    return cv.arcLength(cnt, True)


def get_center(cnt):
    M = cv.moments(cnt)
    c_xy = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
    return c_xy


def get_area(cnt):
    return cv.contourArea(cnt)


def get_width_height(cnt):
    f = [val for sublist in cnt for val in sublist]
    w = max(list(map(lambda x: x[0], f))) - min(list(map(lambda x: x[0], f)))
    h = max(list(map(lambda y: y[1], f))) - min(list(map(lambda y: y[1], f)))
    return w, h