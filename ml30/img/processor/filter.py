import cv2
import numpy as np


class ConvFilter(object):

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src):
        return cv2.filter2D(src, -1, self._kernel)


class SharpenFilter(ConvFilter):
    """ Sharpen filter with a 1-pixel radius."""

    def __init__(self):
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 9, -1],
             [-1, -1, -1]]
        )  # kernel sum adds to 1
        ConvFilter.__init__(self, kernel)


class EdgeFinder(ConvFilter):
    """An edge-finding filter with a 1-pixel radius"""

    def __init__(self):
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]]
        )  # kernel sum adds to 0
        ConvFilter.__init__(self, kernel)


class BlurFilter(ConvFilter):
    """A blur filter with a 2-pixel radius."""

    def __init__(self):
        kernel = np.array(
            [[0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04, 0.04, 0.04, 0.04, 0.04]]
        )
        ConvFilter.__init__(self, kernel)


class EmbossFilter(ConvFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = np.array(
            [[-2, -1, 0],
             [-1, 1, 1],
             [0, 1, 2]]
        )
        ConvFilter.__init__(self, kernel)


class SobelFilter(ConvFilter):

    def __init__(self):
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        ConvFilter.__init__(self, kernel)
