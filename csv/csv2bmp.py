import numpy as np
import sys, getopt
import os

from PIL import Image


def csv2bmp(csvFileName, bmpFileName):
    tempMat = np.genfromtxt(csvFileName, delimiter=";")
    tempMat = tempMat[:, :-1] # Delete last column
    img = Image.fromarray(tempMat) # PIL
    img = img.convert("L")
    img.save(bmpFileName)

def main(argv):
    opts, args = getopt.getopt(argv, "hi:")
    inf = ''
    outf = ''
    for opt, arg in opts:
        if opt == "-i":
            num = arg
    inf = num + ".csv"
    outf = num + ".bmp"
    csv2bmp(inf, outf)

if __name__ == "__main__":
   main(sys.argv[1:])