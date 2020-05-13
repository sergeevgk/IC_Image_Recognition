from PIL import Image, ImageEnhance
import cv2
from utility import gray2rgb, scale_image


def make_rgb_image(name, output_name):
    name1 = name + '_1'
    im = Image.open(name + '.bmp')
    im = ImageEnhance.Contrast(im).enhance(6)
    im = ImageEnhance.Brightness(im).enhance(1.5)
    im.save(name1 + '.bmp')
    image = cv2.imread(name1 + '.bmp', cv2.IMREAD_GRAYSCALE)
    gray2rgb(image)
    image = cv2.imread('1.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.imwrite(output_name, image)
    scale_image(output_name, output_name, width=600, height=480)
    # image = cv2.imread(output_name)
    # return image


def main(name):
    # name = '390'
    output_name = 'data/pseudocolored_images/' + name + '.jpg'
    image = make_rgb_image('csv/' + name, output_name)
    pass


if __name__ == "__main__":
    for i in range(367, 420, 1):
        main(str(i))