import sys
from cv2 import imread, imshow
import cv2
import numpy as np
from detection import detection_alg, detect_hottest_parts
from utility import draw_borders_on_image
from geometry import transform

# main script

def main():
    image_name = 'data/pseudocolored_images/412.jpg'
    image = cv2.imread(image_name)
    hot_parts = detect_hottest_parts(image)
    # draw_borders_on_image(image, hot_parts)
    # решение задачи фотограмметрии для восстановления вида контуров
    hot_part_borders = transform(image, hot_parts)
    # применение нейросети для классификации
    # create black image -> draw borders -> process()
    pass


if __name__ == "__main__":
    main()