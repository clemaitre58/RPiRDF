import numpy as np
from math import factorial
from numba import jit


class ZernikeIndices:

    def __init__(self, i, j):
        self.m = i
        self.n = j


@jit
def _generate_indices():
    tab_ind = []
    tab_ind.append(ZernikeIndices(0, 0))  # 0
    tab_ind.append(ZernikeIndices(-1, 1))  # ...
    tab_ind.append(ZernikeIndices(1, 1))
    tab_ind.append(ZernikeIndices(-2, 2))
    tab_ind.append(ZernikeIndices(0, 2))
    tab_ind.append(ZernikeIndices(2, 2))
    tab_ind.append(ZernikeIndices(-3, 3))
    tab_ind.append(ZernikeIndices(-1, 3))
    tab_ind.append(ZernikeIndices(1, 3))
    tab_ind.append(ZernikeIndices(3, 3))
    tab_ind.append(ZernikeIndices(-4, 4))
    tab_ind.append(ZernikeIndices(-2, 4))
    tab_ind.append(ZernikeIndices(0, 4))
    tab_ind.append(ZernikeIndices(2, 4))
    tab_ind.append(ZernikeIndices(4, 4))  # 15

    return tab_ind


# -------------------------------------------------------------------------
# Copyright C 2015 Gefu Tang
# tanggefu@gmail.com
#
# License Agreement: To acknowledge the use of the code please cite the
#                    following papers:
#
# [1] A. Tahmasbi, F. Saki, S. B. Shokouhi,
#     Classification of Benign and Malignant Masses Based on Zernike Moments,
#     Comput. Biol. Med., vol. 41, no. 8, pp. 726-735, 2011.
#
# [2] F. Saki, A. Tahmasbi, H. Soltanian-Zadeh, S. B. Shokouhi,
#     Fast opposite weight learning rules with application in breast cancer
#     diagnosis, Comput. Biol. Med., vol. 43, no. 1, pp. 32-41, 2013.
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Function to compute Zernike Polynomials:
#
# rad = radialpoly(r,n,m)
# where
#   r = radius
#   n = the order of Zernike polynomial
#   m = the repetition of Zernike moment
# -------------------------------------------------------------------------i

@jit
def _radial_poly(r, n, m):
    rad = np.zeros(r.shape, r.dtype)
    P = (n - abs(m)) / 2
    Q = (n + abs(m)) / 2
    # TODO: if range here = xrange
    for s in range(int(P) + 1):
        c = (-1) ** s * factorial(n - s)
        c /= factorial(s) * factorial(Q - s) * factorial(P - s)
        rad += c * r ** (n - 2 * s)
    return rad


# -------------------------------------------------------------------------
# Function to find the Zernike moments for an N x N binary ROI
#
# Z, A, Phi = Zernikmoment(src, n, m)
# where
#   src = input image
#   n = The order of Zernike moment (scalar)
#   m = The repetition number of Zernike moment (scalar)
# and
#   Z = Complex Zernike moment
#   A = Amplitude of the moment
#   Phi = phase (angle) of the mement (in degrees)
#
# Example:
#   1- calculate the Zernike moment (n,m) for an oval shape,
#   2- rotate the oval shape around its centeroid,
#   3- calculate the Zernike moment (n,m) again,
#   4- the amplitude of the moment (A) should be the same for both images
#   5- the phase (Phi) should be equal to the angle of rotation
# -------------------------------------------------------------------------


@jit
def _zernike_moment(src, n, m):

    H, W = src.shape
    if H > W:
        src = src[(H - W) / 2: (H + W) / 2, :]
    elif H < W:
        src = src[:, (W - H) / 2: (H + W) / 2]

    N = src.shape[0]
    if N % 2:
        src = src[:-1, :-1]
        N -= 1
    x = range(N)
    y = x
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((2 * X - N + 1) ** 2 + (2 * Y - N + 1) ** 2) / N
    Theta = np.arctan2(N - 1 - 2 * Y, 2 * X - N + 1)
    R = np.where(R <= 1, 1, 0) * R

    # get the radial polynomial
    Rad = _radial_poly(R, n, m)

    Product = src * Rad * np.exp(-1j * m * Theta)
    # calculate the moments
    Z = Product.sum()

    # count the number of pixels inside the unit circle
    cnt = np.count_nonzero(R) + 1
    # normalize the amplitude of moments
    Z = (n + 1) * Z / cnt
    # calculate the amplitude of the moment
    A = abs(Z)
    # calculate the phase of the mement (in degrees)
    Phi = np.angle(Z) * 180 / np.pi
    return Z, A, Phi


@jit
def zernike_moment(img, order):
    ind_zer = _generate_indices()

    coef_zernike = []
    for i in range(order):
        Z, A, Phi = _zernike_moment(img, ind_zer[i].n, ind_zer[i].m)
        coef_zernike.append(Z)
    return coef_zernike
