import pathlib
import os
from PIL import Image
import numpy as np
import cv2
import argparse
import tensorflow as tf
import random


def getpaths(path):
    """
    Get all image paths from folder 'path' while avoiding ._ files.
    """
    im_paths = []
    for fil in os.listdir(path):
        if '.png' in fil:
            if "._" in fil:
                # avoid dot underscore
                pass
            else:
                im_paths.append(os.path.join(path, fil))
    return im_paths

def calcmeans(imageFolder):
    """Calculates the mean of a dataset."""
    print("la ruta inicial es ", imageFolder)
    print("esta en el modulo")
    paths = getpaths(imageFolder)
    print("el paht es ", paths)
    total_mean = [0, 0, 0]
    im_counter = 0
    for p in paths:
        print("entro en el path de las imagenes")
        image = np.asarray(Image.open(p))
        mean_rgb = np.mean(image, axis=(0, 1), dtype=np.float64)
        if im_counter % 50 == 0:
            print("Total mean: {} | current mean: {}".format(total_mean, mean_rgb))
            total_mean += mean_rgb
            im_counter += 1
            total_mean /= im_counter
            # rgb to bgr
    #if bgr is True:
    #    total_mean = total_mean[..., ::-1]
    #    print('total media', total_mean)

    print(total_mean)

    return total_mean



