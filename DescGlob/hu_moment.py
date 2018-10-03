from skimage.measure import moments_central
from skimage.measure import moments_hu


def hu_moment(image):
    """hu_moment descriptor for grayscale image.

    Parameters:
        image : ndarray, shape (height, width)
            Input image.

    Returns:
        descriptor : ndarray, shape (7)
            The Hu Momemnt.
    """
    c, r = image.shape

    M = moments_central(image, cr=int(r/2), cc=int(c/2), order=3)
    return moments_hu(M)


def hu_moment_color(image_color):
    """hu_moment descriptor for color image.

    Parameters:
        image : ndarray, shape (height, width, 3)
            Input image.

    Returns:
        descriptor : ndarray, shape (21)
            The Hu Momemnt for each color component.
    """

    M_r = hu_moment(image_color[:, :, 0])
    M_g = hu_moment(image_color[:, :, 1])
    M_b = hu_moment(image_color[:, :, 2])

    M = M_r + M_g + M_b
    return M
