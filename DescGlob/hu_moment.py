from skimage.measure import moments_central
from skimage.measure import moments_hu
import numpy as np


def hu_moment(image):
    """hu_moment descriptor for grayscale image.

    Args
    ----
        image : ndarray, shape (height, width)
            Input image.

    Returns
    -------
        descriptor : ndarray, shape (7)
            The Hu Momemnt.
    """
    c, r = image.shape

    M = moments_central(image, cr=int(r/2), cc=int(c/2), order=3)
    return moments_hu(M)


def hu_moment_color(image_color):
    """hu_moment descriptor for color image.

    Args
    ----
        image : ndarray, shape (height, width, 3)
            Input image.

    Returns
    -------
        descriptor : ndarray, shape (21)
            The Hu Momemnt for each color component.
    """
    r = image_color[:, :, 0]
    g = image_color[:, :, 1]
    b = image_color[:, :, 2]

    M_r = hu_moment(r)
    M_g = hu_moment(g)
    M_b = hu_moment(b)

    return np.hstack([M_r, M_g, M_b])
