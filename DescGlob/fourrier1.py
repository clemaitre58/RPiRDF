import numpy as np


def _norm_fft(in_complex):
    return np.absolute(in_complex)


def _compute_coef_fourrier1(norm):
    h_img = norm.shape[0]
    w_img = norm.shape[1]
    rmax = h_img / 2
    coef = np.zeros(int(rmax))
    coef_fourrier1 = np.zeros(int(rmax)-1)

    for i in range(h_img):
        for j in range(w_img):
            pos_x = j - rmax
            pos_y = i - rmax

            pos_x *= pos_x
            pos_y *= pos_y

            pos = np.sqrt(pos_x + pos_y)

            if pos < rmax and pos > 0:
                ray = int(pos)
                coef[ray] = norm[j][i]
    print(rmax)
    lumi = norm[int(rmax)][int(rmax)]
    print(lumi)

    for i in range(len(coef) - 1):
        coef_fourrier1[i] = coef[i+1] / lumi

    return coef_fourrier1


def fourrier1(in_image):
    # TODO : check if image is square a power of 2
    # TODO : if not crop de image from the center
    # TODO : place the default size to crop
    in_image_fft = np.fft.fft2(in_image)
    # TODO : Normalize FFT2 results?
    in_image_fftshift = np.fft.fftshift(in_image_fft)
    in_fft_norm = _norm_fft(in_image_fftshift)

    desc_fourrier1 = _compute_coef_fourrier1(in_fft_norm)

    return desc_fourrier1
