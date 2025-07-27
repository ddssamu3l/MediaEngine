"""
Utility functions for RIFE frame interpolation.
"""

import cv2
import numpy as np

def read_img(path):
    """Read image from file."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def write_img(path, img):
    """Write image to file."""
    return cv2.imwrite(path, img)
