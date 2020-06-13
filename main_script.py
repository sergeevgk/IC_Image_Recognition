import os
import sys
import cv2
import numpy as np
from detection import detection_alg, detect_hottest_parts, detect_other_parts
from detection import detect_test
from utility import draw_borders_on_image, save_distinct_parts
from visualisation import visualise
from sgd_classifier import SgdClassifier
from cnn_classifier import CnnClassifier
# from prepare_dataset import prepare_data


# main script

def main(num):
    image_name = 'data/gray_images/' + str(num) + '.jpg'
    fig_name = 'data/res/' + str(num) + '.jpg'
    if os.path.isfile(image_name):
        image = cv2.imread(image_name)
        # binary_images, parts = detect_other_parts(image, fig_name)
        binary_images, parts = detect_hottest_parts(image, fig_name)
        # draw_borders_on_image(image, hot_parts)

        # сохранить результаты
        path = "data/test/"
        save_distinct_parts(binary_images, parts, path)
        # работа с классификатором
        classifier = SgdClassifier()
        # classifier = CnnClassifier()
        classifier.train()
        labels = classifier.test()
        visualise(image, parts, labels)
    pass


if __name__ == "__main__":
    for i in range(367, 419, 2):
        main(i)
