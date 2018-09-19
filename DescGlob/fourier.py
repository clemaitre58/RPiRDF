import numpy as np
from numba import jit


@jit
def _compute_coef_fourier1(norm):
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


@jit
def _is_power_2(number):
    isPowOfTwo = True
    while number != 1 and number > 0:
        if number % 2:
            isPowOfTwo = False

        number = number / 2

    return isPowOfTwo and (number > 0)


@jit
def _closest_power_2(num):
    if num > 1:
        for i in range(1, int(num)):
            if (2 ** i > num):
                return 2 ** (i - 1)
    else:
        return 1


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
    h_img, w_img = image.shape
    is_h_img_pow_2 = _is_power_2(h_img)
    is_w_img_pow_2 = _is_power_2(w_img)

    if (is_h_img_pow_2 and is_w_img_pow_2 and (h_img != w_img)) is False:
        # TODO crop the matrix from the center with w and h tcrop he closest
        # power of 2
        if h_img < w_img:
            h_w_size = _closest_power_2(h_img)
        else:
            h_w_size = _closest_power_2(w_img)
    in_image_fft = np.fft.fft2(image)
    # TODO : Normalize FFT2 results?
    in_image_fftshift = np.fft.fftshift(in_image_fft)
    in_fft_norm = np.absolute(in_image_fftshift)
    return _compute_coef_fourier1(in_fft_norm)
