import cv2 as cv
import numpy as np
import PIL.Image
import math
import os


def calculate_median_pixel(pix_array):
    res = np.array([0,0,0])
    for pix in pix_array:
        res = res + pix
    res = (1 / pix_array.shape[0]) * res
    res = map(lambda x: math.ceil(x), res)
    return res


def filter_white_pixels(image, ksize):
    image_copy = image.copy()
    h = image.shape[0]
    w = image.shape[1]
    for i in range(h):
        for j in range(ksize, w - ksize):
            if (image_copy[i,j] == np.array([255, 255, 255])).all():
                image_copy[i,j] = calculate_median_pixel(image_copy[i, j-ksize:j+ksize])
    return image_copy


def gray2rgb(image):
    a = 255
    b = (1.85 * np.pi) / 255
    c = np.pi / 6

    # create empty numpy array needed by the lookup tables
    reds = np.array([])
    greens = np.array([])
    blues = np.array([])

    # pre-compute and assign computed values in the lookup table for each channel
    for i in np.arange(0, 256):
        bx = b * i

        # perform transformation on the r channel: R = a | sin(bx) |
        red = a * np.absolute(np.sin(bx))

        # perform transformation on the g channel: G = a | sin(bx + c) |
        green = a * np.absolute(np.sin(bx + c))

        # perform transformation on the b channel: B = a | sin(bx + 2c) |
        blue = a * np.absolute(np.sin(bx + (2 * c)))

        # append to the numpy array
        reds = np.append(reds, [red])
        greens = np.append(greens, [green])
        blues = np.append(blues, [blue])

    # apply lookup table each matrix: red, green and blue
    r_channel = cv.LUT(image.copy(), reds)
    g_channel = cv.LUT(image.copy(), greens)
    b_channel = cv.LUT(image.copy(), blues)

    # merge the channels
    colored = cv.merge([
        b_channel,
        g_channel,
        r_channel
    ])
    cv.imwrite("1.jpg", colored)
    return colored


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
    cv.imshow("hui", image)
    cv.waitKey()


def rotate_figure(figure, alpha):
    fig = []

    for i in range(len(figure)):
        point = figure[i]
        qx = np.cos(alpha) * point[0] - np.sin(alpha) * point[1]
        qy = np.sin(alpha) * point[0] + np.cos(alpha) * point[1]
        fig.append((qy, qx))
    return fig


def scale_figure(f, scale):
    res = []
    for i in range(len(f)):
        p = f[i]
        res.append((p[0] * scale, p[1] * scale))
    return res


def shift_figure(f, shift):
    res = []
    for i in range(len(f)):
        p = f[i]
        res.append((p[0] + shift[0], p[1] + shift[1]))
    return res


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx**2 + dy**2)


def compare_figures(fig1, fig2):
    f1 = fig1.copy()
    f2 = fig2.copy()

    s = 0
    for i in range(len(f1)):
        min_d = np.Inf
        idx = 0
        for j in range(len(f2)):
            d = dist(f1[i], f2[j])
            if min_d > d:
                min_d = d
                idx = j
        f2.pop(idx)
        s += min_d
    return s