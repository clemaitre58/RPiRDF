from skimage.measure import moments_central
from skimage.measure import moments_hu


def hu_moment(image):
    """hu_moment descriptor.

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
