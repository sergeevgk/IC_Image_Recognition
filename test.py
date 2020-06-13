from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_metrics():
    dir = 'data/gray_images/*.jpg'
    dict = {}
    vals = []
    for file_count, file_name in enumerate(sorted(glob(dir), key=len)):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # vals.append(round(np.mean(img)))
        vals.append(round(np.var(img)))
    dict = {i: v for i,v in enumerate(vals)}
    vv = np.array(vals, dtype=int)
    print(dict)
    plt.plot(vv)
    plt.show()
    pass


if __name__ == "__main__":
    test_metrics()