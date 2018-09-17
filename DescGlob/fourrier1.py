import numpy as np


def _norm_fft(in_complex):
    return np.absolute(in_complex)


def fourrier1(in_image):
    in_image_fft = np.fft.fft2(in_image)
    in_fft_norm = _norm_fft(in_image_fft)

    return in_fft_norm
