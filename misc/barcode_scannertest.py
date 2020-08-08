import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from os import listdir
from os.path import join as pjoin

def decode(img_path):

    image = cv2.imread(img_path)
    decodedObjects = pyzbar.decode(image)
    for obj in decodedObjects:
        print("Type:", obj.type)
        print("Data: ", obj.data, "\n")

    # cv2.imshow("Frame", image)
    # cv2.waitKey(0)

def test(folderpath):
    for each in listdir(folderpath):
        print(each)
        decode(pjoin(folderpath,each))
    print("________________")

# test(folderpath="../data/barcode_samples")