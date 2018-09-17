)

from ..DescGlob import fourrier1
from skimage import io
import matplotlib.pyplot as plt


def main():
    img = io.imread('lena128x128.jpg')
    # io.imshow(img)
    # io.show()
    # print(img.shape)
    norm_four = fourrier1(img)
    plt.imshow(norm_four)
    plt.show()


if __name__ == '__main__':
    main()
