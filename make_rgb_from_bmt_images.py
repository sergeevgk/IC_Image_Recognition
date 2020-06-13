from PIL import Image, ImageEnhance
import cv2 as cv
from utility import scale_image
import numpy as np


def make_rgb_image(name, output_name, gray_output_name):
    im = Image.open(name + '.bmp')
    im = ImageEnhance.Contrast(im).enhance(6)
    im = ImageEnhance.Brightness(im).enhance(1.5)
    im.save(gray_output_name)
    scale_image(gray_output_name, gray_output_name, width=600, height=480)
    image = cv.imread(gray_output_name, cv.IMREAD_GRAYSCALE)
    gray2rgb(image)
    image = cv.imread('temp.jpg')
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    image = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    cv.imwrite(output_name, image)
    scale_image(output_name, output_name, width=600, height=480)
    # image = cv.imread(output_name)
    # return image


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
    cv.imwrite("temp.jpg", colored)
    return colored


def main(name):
    output_name = 'data/pseudocolored_images/' + name + '.jpg'
    gray_output_name = 'data/gray_images/' + name + '.jpg'
    make_rgb_image('csv/' + name, output_name, gray_output_name)
    pass


if __name__ == "__main__":
    for i in range(367, 420, 1):
        main(str(i))