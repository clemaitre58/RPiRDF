import numpy as np


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
    # lumi = norm[rmax][rmax]
    lumi = 1
    for i in range(len(coef) - 1):
        coef_fourier1[i] = coef[i + 1] / lumi

    return coef_fourier1


def _is_power_2(number):
    isPowOfTwo = True
    while number != 1 and number > 0:
        if number % 2:
            isPowOfTwo = False

        number = number / 2

    return isPowOfTwo and (number > 0)


def _closest_power_2(num):
    if num > 1:
        for i in range(1, int(num)):
            if (2 ** i > num):
                return 2 ** (i - 1)
    else:
        return 1


def _crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty+cropy, startx:startx+cropx]


def fourier1(image):
    """Fourier descriptor.

    Args
    ----
        image : ndarray, shape (height, width)
            Input image.

    Returns
    -------
        descriptor : ndarray, shape (height // 2 -1,)
            The Fourier descriptor.
    """

    h_img, w_img = image.shape
    is_h_img_pow_2 = _is_power_2(h_img)
    is_w_img_pow_2 = _is_power_2(w_img)

    if (is_h_img_pow_2 and is_w_img_pow_2 and (h_img != w_img)) is False:
        if h_img < w_img:
            h_w_size = _closest_power_2(h_img)
        else:
            h_w_size = _closest_power_2(w_img)

        image_crop = _crop_center(image, h_w_size, h_w_size)
        image = image_crop

    in_image_fft = np.fft.fft2(image)
    # TODO : Normalize FFT2 results?
    in_image_fftshift = np.fft.fftshift(in_image_fft)
    in_fft_norm = np.absolute(in_image_fftshift)
    return _compute_coef_fourier1(in_fft_norm)


def fourier1_color(image_color):
    """Fourier descriptor.

    Args
    ----
        image : ndarray, shape (height, width)
            Input color image.

    Returns
    -------
        descriptor : ndarray, shape ((height // 2 -1 * )3,)
            The Fourier descriptor.
    """
    r = image_color[:, :, 0]
    g = image_color[:, :, 1]
    b = image_color[:, :, 2]

    f_r = fourier1(r)
    f_g = fourier1(g)
    f_b = fourier1(b)

    return np.hstack([f_r, f_g, f_b])
