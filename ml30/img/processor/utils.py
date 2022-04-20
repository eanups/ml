
import cv2
import numpy as np
from scipy import ndimage
import scipy.interpolate
import matplotlib.pyplot as plt
from enum import Enum


class Color(Enum):
    BLUE = 0
    GREEN = 1
    RED = 2


class Utils:
    @staticmethod
    def show_img(img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

