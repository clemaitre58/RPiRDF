import numpy as np
from numba import jit


@jit
def _compute_coef_fourrier1(norm):
    h_img = norm.shape[0]
    w_img = norm.shape[1]
    rmax = h_img // 2
    coef = np.zeros(rmax, dtype=np.float64)
    coef_fourier1 = np.zeros(rmax - 1, dtype=np.float64)

    for i in range(h_img):
        for j in range(w_img):
            pos_x = (j - rmax) ** 2
            pos_y = (i - rmax) ** 2
            pos = np.sqrt(pos_x + pos_y)

            if 0 < pos < rmax:
                ray = int(pos)
                coef[ray] = norm[j][i]
    lumi = norm[rmax][rmax]
    for i in range(len(coef) - 1):
        coef_fourier1[i] = coef[i + 1] / lumi

    return coef_fourier1


def fourier1(image):
    """Fourier descriptor.

    Parameters:
        image : ndarray, shape (height, width)
            Input image.

    Returns:
        descriptor : ndarray, shape (height // 2 -1,)
            The Fourier descriptor.
    """

    # TODO : check if image is square a power of 2
    # TODO : if not crop de image from the center
    # TODO : place the default size to crop
    in_image_fft = np.fft.fft2(image)
    # TODO : Normalize FFT2 results?
    in_image_fftshift = np.fft.fftshift(in_image_fft)
    in_fft_norm = np.absolute(in_image_fftshift)
    return _compute_coef_fourrier1(in_fft_norm)
